from PIL import Image
from torchvision import transforms

from collections import namedtuple, deque
import random

from .simple_dqn import SimpleDQN


model_name_to_model = {
    'simple_dqn': SimpleDQN
}

def model_transforms(model_name, dtype):
    # simple_dqn
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(40, interpolation=Image.CUBIC),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype)
    ])

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
