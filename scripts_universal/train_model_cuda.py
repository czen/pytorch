import os
# chdir is required before importing torch, since otherwise
# "python scripts_universal/train_model_cuda.py" won't work in install mode
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime, timedelta
import logging
import pathlib
import random
import sys
from time import time

from common.disclaimer import torchvision_torchmetrics_disclaimer

import torch
import numpy as np

try:
    import torchvision
    import torchmetrics
except ModuleNotFoundError:
    print(torchvision_torchmetrics_disclaimer)
    sys.exit(0)

from common.models import model_name_to_model, model_transforms

def main():
    os.makedirs('data/train_model_cuda', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'data/train_model_cuda/log_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log',
        filemode='a+',
        format='%(asctime)-15s %(levelname)-8s %(message)s'
    )
    logging.info(f'Running {" ".join(sys.argv)}')
    # Output logs to file and to stderr
    logging.getLogger().addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser(
        description=(
            'Trains a model with dtype=torch.float32 using CUDA, '
            'writes the snapshots and the results into the '
            'data/train_model_cuda/ folder.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_name', type=str, default='simple', choices=['simple', 'mobilenetv3', 'convnext'],
        help='the model to train and test'
    )
    parser.add_argument(
        '--test-only', dest='test_only', default=False, action='store_true',
        help='do not train the model, only load and test it'
    )
    parser.add_argument(
        '--epochs', default=2, type=int,
        help='number of epochs for training'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size', default=6, type=int,
        help='batch size'
    )
    parser.add_argument(
        '--random', action='store_true',
        help='randomize the training'
    )

    args = parser.parse_args()

    if not args.random:
        # Make the result reproducible
        random_seed = 42
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)

        # Random number generators for DataLoader
        def new_generator():
            g = torch.Generator()
            g.manual_seed(random_seed)
            return g

        # Seed the parallel workers in DataLoader
        def seed_worker(worker_id):
            worker_seed = random_seed + worker_id
            torch.manual_seed(worker_seed)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        def new_generator():
            return torch.Generator()

        def seed_worker(worker_id):
            pass

    model_name = args.model_name
    model = model_name_to_model[model_name]
    transform = model_transforms(model_name, dtype)
    epochs = args.epochs
    if epochs <= 0:
        logging.error('--epochs has to be >= 1')
    batch_size = args.batch_size
    if batch_size <= 0:
        logging.error('--batch-size has to be >= 1')

    dtype = torch.float32

    model_file_name = f'data/test_train_with_universal_types/{model_name}_{{:05d}}.pth'

    # Initialize CIFAR10
    trainset = torchvision.datasets.CIFAR10(
        root='data/CIFAR10', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2,
        generator=new_generator(),
        worker_init_fn=seed_worker
    )

    testset = torchvision.datasets.CIFAR10(
        root='data/CIFAR10', train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2,
        generator=new_generator(),
        worker_init_fn=seed_worker
    )

    # Create the model
    net = model(num_classes=10).to(dtype).train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    # Train for only one epoch since training on CPU is very slow
    logging.info(f'Training {model_name} with {dtype} (steps per epoch: {len(trainloader)})')
    train_start_time = time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                logging.info(
                    f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}'
                )
                running_loss = 0.0
        # Save the snapshot
        torch.save({
            'epoch': epoch,
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_file_name.format(epoch))
    train_end_time = time()
    train_time = timedelta(seconds=train_end_time - train_start_time)
    logging.info(f'Done. Time: {train_time}')

if __name__ == '__main__':
    main()
