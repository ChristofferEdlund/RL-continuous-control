import torch
import torch.nn.functional as F

import numpy as np
from model import Critic, Actor
import random
from collections import namedtuple, deque
import copy

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128 #64 #32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-3 #1e-4 #5e-4  # learning rate for the actor network
LR_CRITIC = 1e-3 #5e-4 # learning rate fot the critic network
UPDATE_EVERY = 20  # how often to update the network
NUM_UPDATES = 10 # How many updates to make
NOISE_EPSILON = 1
NOISE_DECAY = 0.9999
# WEIGHT_DECAY = 1e-2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """
    My RL agent that learns and interact with the enviornment.
    I will try to implement a DDPG algorithm here
    """
    def __init__(self, state_size, action_size, seed, use_noise=True, grad_clip=True):
        """
        Inits the DDPG Agent

        :param state_size: - int size of the state vector
        :param action_size: int size of the action vector
        :param seed: random seed for agent
        """

        self.seed = seed
        random.seed(self.seed)

        ## Generate target and local critic and actor networks
        self.local_actor = Actor(state_size, action_size, self.seed).to(device)
        self.target_actor = Actor(state_size, action_size, self.seed).to(device)

        self.local_critic = Critic(state_size, action_size, self.seed).to(device)
        self.target_critic = Critic(state_size, action_size, self.seed).to(device)

        ## We need optimzers for the local networks, the taget networks will be updated via soft-updates
        # from their local counterparts.
        self.actor_optimizer = torch.optim.Adam(self.local_actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.local_critic.parameters(), lr=LR_CRITIC)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.use_noise = use_noise
        self.noise_epsilon = NOISE_EPSILON
        self.noise = OUNoise(action_size, seed)
        self.grad_clip = grad_clip
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, time=None):
        # Save experience in replay memory

        self.memory.add(state, action, reward, next_state, done)

        if time is None:
            time = self.t_step

        self.t_step = (time + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            for update in range(NUM_UPDATES):
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

            self.noise_epsilon *= NOISE_DECAY


    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()

        if self.use_noise:
            action = action + self.noise.sample() * self.noise_epsilon

        return np.clip(action, -1, 1) # <- action space bound to [-1, 1]


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update the critic network
        next_actions = self.target_actor(next_states)
        pred_future_Q_values = self.target_critic(next_states, next_actions)
        # Compute the Q targets
        target_Q = rewards + (gamma * pred_future_Q_values * (1-dones))
        # Critic loss
        pred_Q = self.local_critic(states, actions)
        loss_critic = F.mse_loss(pred_Q, target_Q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), 1) # <- Gradient clipping
        self.critic_optimizer.step()

        # Update actor network

        pred_action = self.local_actor(states)
        loss_actor = -self.local_critic(states, pred_action).mean() # Make it negative since we want to maximize the reward

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()


        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.local_critic, self.target_critic, TAU)
        self.soft_update(self.local_actor, self.target_actor, TAU)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def reset_noise(self):
        self.noise.reset()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Code taken from: https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state