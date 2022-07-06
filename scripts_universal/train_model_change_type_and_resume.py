import os
# chdir is required before importing torch, since otherwise
# "python scripts_universal/train_model_change_type_and_resume.py" won't work in install mode
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

def optimizer_to(optimizer, target):
    for param in optimizer.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(target)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(target)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(target)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(target)

dtype_name_to_dtype = {
    'cfloatwithsubnormals': torch.float32
}

def main():
    os.makedirs('data/train_model_change_type_and_resume', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'data/train_model_change_type_and_resume/log_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log',
        filemode='a+',
        format='%(asctime)-15s %(levelname)-8s %(message)s'
    )
    logging.info(f'Running {" ".join(sys.argv)}')
    # Output logs to file and to stderr
    logging.getLogger().addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser(
        description=(
            'Loads a model snapshot that was created by train_model_cuda.py, '
            'converts it to the specified type, and trains it for one more '
            'epoch.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'snapshot', type=str,
        help='the snapshot file (*.pth)'
    )
    parser.add_argument(
        '--dtype', default='cfloatwithsubnormals', choices=['cfloatwithsubnormals'],
        help='batch size'
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

    if args.dtype not in dtype_name_to_dtype:
        logging.error(f'Invalid dtype: {args.dtype}')
        return

    dtype = dtype_name_to_dtype[args.dtype]
    batch_size = args.batch_size
    if batch_size <= 0:
        logging.error('--batch-size has to be >= 1')

    snapshot = torch.load(args.snapshot)

    if 'model_name' not in snapshot:
        logging.error('Invalid snapshot. Please specify a snapshot created by train_model_cuda.py.')
        return
    model_name = snapshot['model_name']

    model = model_name_to_model[model_name]
    transform = model_transforms(model_name, dtype)

    output_snapshot_name = pathlib.Path(args.snapshot)
    output_snapshot_name = output_snapshot_name.with_name(
        output_snapshot_name.stem + '_resumed' + output_snapshot_name.suffix
    )

    epoch = snapshot['epoch']

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
    net = model(num_classes=10).to(torch.float64)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    net.load_state_dict(snapshot['model_state_dict'])
    optimizer.load_state_dict(snapshot['optimizer_state_dict'])

    # Convert the model to target dtype
    net = net.to(dtype)
    optimizer_to(optimizer, dtype)

    net = net.train()

    running_loss = 0

    logging.info(f'Training {model_name} with {dtype} (steps per epoch: {len(trainloader)})')
    train_start_time = time()
    train_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        running_loss += loss.item()
        if i % 100 == 99:
            logging.info(
                f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}'
            )
            running_loss = 0.0
    train_end_time = time()
    train_time = timedelta(seconds=train_end_time - train_start_time)
    train_loss /= len(trainloader.sampler)
    logging.info(f'Done. Loss: {train_loss}, time: {train_time}')

    # Save the model
    torch.save({
        'model_name': model_name,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, str(output_snapshot_name))

    # Test the net
    # Calculate accuracy, MAP and MAR
    accuracy = torchmetrics.Accuracy()
    precision = torchmetrics.Precision(num_classes=10, average='macro')
    recall = torchmetrics.Recall(num_classes=10, average='macro')
    logging.info(f'Testing {model_name} with {dtype} (steps: {len(testloader)})')
    net = net.eval()
    test_start_time = time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            accuracy(labels, predicted)
            precision(labels, predicted)
            recall(labels, predicted)
    test_end_time = time()
    test_time = timedelta(seconds=test_end_time - test_start_time)

    logging.info(
        f'Done. Time: {test_time}, accuracy: {accuracy.compute().item()}%, '
        f'macro average precision: {precision.compute().item()}, '
        f'macro average recall: {recall.compute().item()}'
    )

if __name__ == '__main__':
    main()
