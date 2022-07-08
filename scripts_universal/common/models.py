import torchvision
from torchvision import transforms

from .simple_net import SimpleNet

model_name_to_model = {
    'mobilenetv3': torchvision.models.mobilenet_v3_small,
    'convnext': torchvision.models.convnext_base,
    'simple': SimpleNet
}

def model_transforms(model_name, dtype):
    if model_name == 'simple':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif model_name == 'mobilenetv3':
        return transforms.Compose([
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
        return transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
