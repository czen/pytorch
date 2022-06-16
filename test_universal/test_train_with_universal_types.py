import os
# chdir is required before importing torch, since otherwise
# "python test_universal/test_train_with_universal_types.py" won't work in install mode
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import argparse
import csv
from datetime import timedelta
import logging
import random
from time import time


class SimpleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
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

    parser = argparse.ArgumentParser(
        description=(
            'Trains and/or tests a model with dtype=torch.cfloatwithsubnormals '
            'and dtype=torch.float32, writes the models and the results into '
            'the test_data/test_train_with_universal_types/ folder. '
            'This script requires torchvision. Install torchvision first, and '
            'then reinstall the modified pytorch (python setup.py install).'
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

    args = parser.parse_args()

    model_name = args.model_name
    model_name_to_model = {
        'mobilenetv3': torchvision.models.mobilenet_v3_small,
        'convnext': torchvision.models.convnext_base,
        'simple': SimpleNet,
    }
    model = model_name_to_model[model_name]
    test_only = args.test_only
    epochs = args.epochs
    if epochs <= 0:
        logging.error('--epochs has to be >= 1')
    batch_size = args.batch_size
    if batch_size <= 0:
        logging.error('--batch-size has to be >= 1')

    os.makedirs('test_data/test_train_with_universal_types', exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'test_data/test_train_with_universal_types/{model_name}_log.log',
        filemode='a+',
        format='%(asctime)-15s %(levelname)-8s %(message)s'
    )
    # Output logs to file and to stderr
    logging.getLogger().addHandler(logging.StreamHandler())

    dtypes = (torch.cfloatwithsubnormals, torch.float32)

    for dtype in dtypes:
        model_file_name = f'test_data/test_train_with_universal_types/{model_name}_small_{dtype}.pth'

        # Initialize CIFAR10
        if model_name == 'simple':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif model_name == 'mobilenetv3':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:  # convnext
            transform = transforms.Compose([
                transforms.Resize(232),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        trainset = torchvision.datasets.CIFAR10(
            root='test_data/CIFAR10', train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2,
            generator=new_generator(),
            worker_init_fn=seed_worker
        )

        testset = torchvision.datasets.CIFAR10(
            root='test_data/CIFAR10', train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2,
            generator=new_generator(),
            worker_init_fn=seed_worker
        )

        if not test_only:
            # Create mobilenetv3
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
            train_end_time = time()
            train_time = timedelta(seconds=train_end_time - train_start_time)
            logging.info(f'Done. Time: {train_time}')
            # Save model
            torch.save(net.state_dict(), model_file_name)
        # Load model
        if not os.path.isfile(model_file_name):
            logging.error(f'{model_file_name} does not exist')
            return
        net = model(num_classes=10).to(dtype)
        net.load_state_dict(torch.load(model_file_name))
        net = net.eval()

        # Test the net
        # Calculate accuracy, MAP and MAR manually to avoid adding more dependecies
        correct = 0
        total = 0
        true_positives_per_class = [0] * 10
        true_negatives_per_class = [0] * 10
        false_positives_per_class = [0] * 10
        false_negatives_per_class = [0] * 10
        logging.info(f'Testing {model_name} with {dtype} (steps: {len(testloader)})')
        test_start_time = time()
        with open(f'test_data/test_train_with_universal_types/{model_name}_{dtype}_test_result.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Prediction', 'Label'])
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    for prediction, label in zip(predicted, labels):
                        for class_i in range(10):
                            prediction_bin = (prediction == class_i)
                            label_bin = (label == class_i)
                            if prediction_bin and label_bin:
                                true_positives_per_class[class_i] += 1
                            elif not prediction_bin and label_bin:
                                false_negatives_per_class[class_i] += 1
                            elif prediction_bin and not label_bin:
                                false_positives_per_class[class_i] += 1
                            else:
                                true_negatives_per_class[class_i] += 1
                        writer.writerow([prediction.item(), label.item()])
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        test_end_time = time()
        test_time = timedelta(seconds=test_end_time - test_start_time)
        accuracy = 100 * correct // total
        macro_average_precision = 0
        macro_average_recall = 0
        for class_i in range(10):
            macro_average_precision += (
                true_positives_per_class[class_i] /
                (true_positives_per_class[class_i] + false_positives_per_class[class_i])
            )
            macro_average_recall += (
                true_positives_per_class[class_i] /
                (true_positives_per_class[class_i] + false_negatives_per_class[class_i])
            )
        macro_average_precision /= 10
        macro_average_recall /= 10

        logging.info(
            f'Done. Time: {test_time}, accuracy: {accuracy}%, '
            f'macro average precision: {macro_average_precision}, '
            f'macro average recall: {macro_average_recall}'
        )

if __name__ == '__main__':
    main()
