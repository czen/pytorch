import os
# chdir is required before importing torch, since otherwise
# "python scripts_universal/train_model_cuda.py" won't work in install mode
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime, timedelta
import logging
import random
import re
import sys
from time import time

from common.disclaimer import torchvision_torchmetrics_disclaimer

import torch
from torch.utils.tensorboard import SummaryWriter
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
            'data/train_model_cuda/ folder. Resumes the training from the last '
            'epoch if the training was started and interrupted before.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model-name', dest='model_name', type=str, default='simple', choices=['simple', 'mobilenetv3', 'convnext'],
        help='the model to train'
    )
    parser.add_argument(
        '--train-until', dest='train_until', default=0.75, type=float,
        help='stop training once the specified accuracy is reached (ignored if --epochs is set)'
    )
    parser.add_argument(
        '--epochs', type=int,
        help='train for the specified number of epochs'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size', default=32, type=int,
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

    dtype = torch.float32

    model_name = args.model_name
    model = model_name_to_model[model_name]
    transform = model_transforms(model_name, dtype)
    train_until = args.train_until
    if train_until < 0 or train_until > 1:
        logging.error('--train-until has to between 0 and 1')
        return
    epochs = args.epochs
    if epochs is not None and epochs <= 0:
        logging.error('--epochs has to be >= 1')
        return
    batch_size = args.batch_size
    if batch_size <= 0:
        logging.error('--batch-size has to be >= 1')
        return

    # Initialize CIFAR10
    trainset = torchvision.datasets.CIFAR10(
        root='data/CIFAR10', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2,
        generator=new_generator(),
        worker_init_fn=seed_worker
    )

    valset = torchvision.datasets.CIFAR10(
        root='data/CIFAR10', train=False, download=True, transform=transform
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=2,
        generator=new_generator(),
        worker_init_fn=seed_worker
    )

    # Create the model
    net = model(num_classes=10).to(dtype).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )

    model_file_name = f'data/train_model_cuda/{model_name}_{{:05d}}.pth'

    # Check if there are snapshots
    snapshot_epoch = -1
    snapshot_file_name = None
    for file_name in os.listdir('data/train_model_cuda'):
        if m := re.match(f'{model_name}_([0-9]+).pth', file_name):
            new_snapshot_epoch = int(m.group(1))
            if new_snapshot_epoch > snapshot_epoch:
                snapshot_epoch = new_snapshot_epoch
                snapshot_file_name = file_name

    if snapshot_file_name is not None:
        snapshot = torch.load(os.path.join('data/train_model_cuda', snapshot_file_name))
        epoch = snapshot['epoch'] + 1
        current_accuracy = snapshot['accuracy']
        net.load_state_dict(snapshot['model_state_dict'])
        optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        logging.info(f'Continuing training from epoch {epoch}')
    else:
        epoch = 0
        current_accuracy = 0

    tensorboard_writer = SummaryWriter(
        log_dir=f'data/train_model_cuda/tensorboard_{model_name}'
    )

    # Train for only one epoch since training on CPU is very slow
    logging.info(f'Training {model_name} model with {dtype} (steps per epoch: {len(trainloader)})')
    while True:
        if epochs is not None:
            if epoch >= epochs:
                break
        else:
            if current_accuracy >= train_until:
                break
        net.train()
        train_loss = 0.0
        running_loss = 0.0
        accuracy = torchmetrics.Accuracy().cuda()
        precision = torchmetrics.Precision(num_classes=10, average='macro').cuda()
        recall = torchmetrics.Recall(num_classes=10, average='macro').cuda()
        train_start_time = time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            accuracy(labels, predicted)
            precision(labels, predicted)
            recall(labels, predicted)

            train_loss += loss.item() * labels.size(0)
            running_loss += loss.item()
            if i % 100 == 99:
                logging.info(
                    f'[{epoch}, {i + 1:5d}] loss: {running_loss / 100:.3f}'
                )
                running_loss = 0.0
        train_end_time = time()
        train_time = timedelta(seconds=train_end_time - train_start_time)
        train_loss /= len(trainloader.sampler)

        tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
        tensorboard_writer.add_scalar('Accuracy/train', accuracy.compute().item(), epoch)
        tensorboard_writer.add_scalar('MacroAveragePrecision/train', precision.compute().item(), epoch)
        tensorboard_writer.add_scalar('MacroAverageRecall/train', recall.compute().item(), epoch)

        # Validate the model
        logging.info(f'Validating {model_name} model with {dtype} (steps: {len(valloader)})')
        net = net.eval()
        accuracy = torchmetrics.Accuracy().cuda()
        precision = torchmetrics.Precision(num_classes=10, average='macro').cuda()
        recall = torchmetrics.Recall(num_classes=10, average='macro').cuda()
        val_loss = 0.0
        val_start_time = time()
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                accuracy(labels, predicted)
                precision(labels, predicted)
                recall(labels, predicted)
        val_end_time = time()
        val_time = timedelta(seconds=val_end_time - val_start_time)
        val_loss /= len(valloader.sampler)

        current_accuracy = accuracy.compute().item()
        current_precision = precision.compute().item()
        current_recall = recall.compute().item()

        tensorboard_writer.add_scalar('Loss/validation', val_loss, epoch)
        tensorboard_writer.add_scalar('Accuracy/validation', current_accuracy, epoch)
        tensorboard_writer.add_scalar('MacroAveragePrecision/validation', current_precision, epoch)
        tensorboard_writer.add_scalar('MacroAverageRecall/validation', current_recall, epoch)

        # Save the snapshot
        torch.save({
            'epoch': epoch,
            'accuracy': current_accuracy,
            'precision': current_precision,
            'recall': current_recall,
            'model_name': model_name,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_file_name.format(epoch))
        tensorboard_writer.flush()

        epoch += 1

        logging.info(
            f'Epoch done. Time: {val_time}, accuracy: {current_accuracy}%, '
            f'macro average precision: {precision.compute().item()}, '
            f'macro average recall: {recall.compute().item()}'
        )
    logging.info('Done')

if __name__ == '__main__':
    main()
