import os
# chdir is required before importing torch, since otherwise
# "python scripts_universal/train_rl_model_cuda.py" won't work in install mode
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
from datetime import datetime, timedelta
from itertools import count
import logging
import math
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

try:
    import gym
except ModuleNotFoundError:
    print(
        'This script requires gym. '
        'Please install it with "pip install gym[classic_control]" or '
        '"conda install -c conda-forge gym[classic_control]".'
    )
    sys.exit(0)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(
        'This script requires matplotlib. '
        'Please install it with "pip install matplotlib" or '
        '"conda install -c conda-forge matplotlib".'
    )
    sys.exit(0)

from common.rl_models import model_name_to_model, model_transforms, Transition, ReplayMemory

def get_cart_location(env, screen):
    world_width = env.x_threshold * 2
    screen_width = screen.shape[-1]
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen(env, transform):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return transform(screen).unsqueeze(0)

def select_action(env, policy_net, state, steps_done, eps_start, eps_end, eps_decay):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device='cuda', dtype=torch.long)

def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model(policy_net, target_net, criterion, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(
        lambda s: s is not None,
        batch.next_state
    )), device='cuda', dtype=torch.bool)
    non_final_next_states = torch.cat([
        s for s in batch.next_state if s is not None
    ])
    state_batch = torch.cat(batch.state).cuda()
    action_batch = torch.cat(batch.action).cuda()
    reward_batch = torch.cat(batch.reward).cuda()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device='cuda')
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main():
    os.makedirs('data/train_rl_model_cuda', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f'data/train_rl_model_cuda/log_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log',
        filemode='a+',
        format='%(asctime)-15s %(levelname)-8s %(message)s'
    )
    logging.info(f'Running {" ".join(sys.argv)}')
    # Output logs to file and to stderr
    logging.getLogger().addHandler(logging.StreamHandler())

    parser = argparse.ArgumentParser(
        description=(
            'Trains a reinforcement learning model with dtype=torch.float32 '
            'using CUDA, writes the snapshots and the results into the '
            'data/train_rl_model_cuda/ folder. Resumes the training from the last '
            'epoch if the training was started and interrupted before.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model-name', dest='model_name', type=str, default='simple_dqn', choices=['simple_dqn'],
        help='the model to train'
    )
    parser.add_argument(
        '--episodes', default=50, type=int,
        help='train for the specified number of episodes'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size', default=128, type=int,
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

    episodes = args.episodes
    if episodes <= 0:
        logging.error('--episodes has to be >= 1')
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

    # Create the environment
    env = gym.make('CartPole-v0').unwrapped
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10

    # Get screen dimensions
    plt.ion()
    env.reset()
    init_screen = get_screen(env, transform)
    screen_height, screen_width = init_screen.shape[-2:]

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    # Create the model
    policy_net = model(screen_height, screen_width, n_actions).cuda()

    optimizer = torch.optim.RMSprop(policy_net.parameters())
    criterion = torch.nn.SmoothL1Loss()
    memory = ReplayMemory(10000)

    model_file_name = f'data/train_rl_model_cuda/{model_name}_{{:05d}}.pth'

    # Check if there are snapshots
    snapshot_epoch = -1
    snapshot_file_name = None
    for file_name in os.listdir('data/train_rl_model_cuda'):
        if m := re.match(f'{model_name}_([0-9]+).pth', file_name):
            new_snapshot_epoch = int(m.group(1))
            if new_snapshot_epoch > snapshot_epoch:
                snapshot_epoch = new_snapshot_epoch
                snapshot_file_name = file_name

    # if snapshot_file_name is not None:
    #     snapshot = torch.load(os.path.join('data/train_rl_model_cuda', snapshot_file_name))
    #     epoch = snapshot['epoch'] + 1
    #     current_accuracy = snapshot['accuracy']
    #     net.load_state_dict(snapshot['model_state_dict'])
    #     optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    #     logging.info(f'Continuing training from epoch {epoch}')
    # else:
    #     epoch = 0
    #     current_accuracy = 0

    policy_net.train()
    target_net = model(screen_height, screen_width, n_actions).cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    tensorboard_writer = SummaryWriter(
        log_dir=f'data/train_rl_model_cuda/tensorboard_{model_name}'
    )

    logging.info(f'Training {model_name} model with {dtype} (steps per epoch: {len(trainloader)})')
    steps_done = 0
    episode_durations = []
    for i_episode in range(episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, transform).cuda()
        current_screen = get_screen(env, transform).cuda()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = select_action(env, policy_net, state, steps_done, eps_start, eps_end, eps_decay)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device='cuda')

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, transform).cuda()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_net, target_net, criterion, optimizer, memory, batch_size, gamma)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # tensorboard_writer.add_scalar('Loss/train', train_loss, epoch)
        # tensorboard_writer.add_scalar('Accuracy/train', accuracy.compute().item(), epoch)
        # tensorboard_writer.add_scalar('MacroAveragePrecision/train', precision.compute().item(), epoch)
        # tensorboard_writer.add_scalar('MacroAverageRecall/train', recall.compute().item(), epoch)

    env.render()
    env.close()
    plt.ioff()
    plt.show()

    logging.info('Done')

if __name__ == '__main__':
    main()
