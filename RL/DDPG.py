import gym
import random
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, value_dim=1):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, value_dim))
    
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim),
                                 nn.Tanh())
        self.action_bound = action_bound
    
    def forward(self, x):
        return self.net(x) * self.action_bound

class ReplayBuffer:
    def __init__(self, max_len):
        self.buffer = collections.deque(maxlen=max_len)
    
    def add(self, s, a, s_, r, d):
        self.buffer.append((s, a, s_, r, d))
    
    def sample(self, size):
        return zip(*random.sample(self.buffer, size))
    
    @property
    def size(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, tau, sigma, lr_critic, lr_actor, action_bound):
        self.critic = Critic(state_dim, hidden_dim, action_dim)
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_ = Critic(state_dim, hidden_dim, action_dim)
        self.actor_ = Actor(state_dim, hidden_dim, action_dim, action_bound)
        self.critic_.load_state_dict(self.critic.state_dict())
        self.actor_.load_state_dict(self.actor.state_dict())
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.action_dim = action_dim
    
    def take_action(self, s):
        s = torch.tensor(s, dtype=torch.float)
        a = self.actor(s).item()
        a += self.sigma * np.random.randn(self.action_dim)
        return a
    
    def update(self, transition):
        s = torch.tensor(transition['state'], dtype=torch.float)
        s_ = torch.tensor(transition['next_state'], dtype=torch.float)
        r = torch.tensor(transition['reward'], dtype=torch.float)
        a = torch.tensor(transition['action'], dtype=torch.float)
        d = torch.tensor(transition['done'], dtype=torch.int64)

        a_ = self.actor_(s_)
        td_target = r + self.gamma * self.critic_(torch.cat((s_, a_), dim=1)) * (1 - d)
        loss_critic = F.mse_loss(self.critic(torch.cat((s, a), dim=1)), td_target)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        loss_actor = -torch.mean(self.critic(torch.cat((s, self.actor(s)), dim=1)))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_)
        self.soft_update(self.actor, self.actor_)

    def soft_update(self, net, net_):
        for p, p_ in zip(net.parameters(), net_.parameters()):
            p_.data.copy_(self.tau * p.data + (1 - self.tau) * p_.data)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    hidden_dim = 64
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim, hidden_dim, action_dim, gamma=0.98, tau=0.005, sigma=0.01, 
                 lr_critic=3e-3, lr_actor=3e-4, action_bound=env.action_space.high[0])
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    eposides = 200
    returns = np.zeros(eposides)
    replaybuffer = ReplayBuffer(max_len=10000)
    minimal_size = 1000
    batch_size = 64

    for i in tqdm(range(eposides)):
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state.reshape(1, -1))
            next_state, reward, done, info = env.step(action)
            replaybuffer.add(state, action, next_state, reward, done)
            state = next_state
            returns[i] += reward
            if replaybuffer.size < minimal_size:
                continue
            s, a, s_, r, d = replaybuffer.sample(batch_size)
            transition = {'state': np.array(s),
                          'next_state': np.array(s_),
                          'action': np.array(a).reshape(-1, 1),
                          'reward': np.array(r).reshape(-1, 1),
                          'done': np.array(d).reshape(-1, 1)}
            agent.update(transition)
    
    plt.figure()
    plt.plot(range(eposides), returns)
    plt.show()