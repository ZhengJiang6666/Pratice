import gym
import random
import collections
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.fc = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_sigma = nn.Linear(hidden_dim, action_dim)
        self.softplus = nn.Softplus()
        self.action_bound = action_bound
    
    def forward(self, x):
        x = self.relu(self.fc(x))
        mu = self.fc_mu(x)
        sigma = self.softplus(self.fc_sigma(x))
        dist = torch.distributions.Normal(mu, sigma)
        a = dist.rsample()
        log_proba = dist.log_prob(a)
        a = torch.tanh(a)
        log_proba -= torch.log(1 - a.pow(2) + 1e-7)
        return a * self.action_bound, log_proba
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, value_dim=1):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, value_dim))
    
    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = self.net(x)
        return x

class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_actor, lr_critic, lr_alpha,
                 gamma, tau, entropy_bound, action_bound):
        self.critic1 = ValueNet(state_dim, hidden_dim, action_dim)
        self.critic_target1 = ValueNet(state_dim, hidden_dim, action_dim)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic2 = ValueNet(state_dim, hidden_dim, action_dim)
        self.critic_target2 = ValueNet(state_dim, hidden_dim, action_dim)
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)

        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr_critic)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr_actor)

        self.alpha = torch.log(torch.tensor(0.01))
        self.alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.alpha], lr_alpha)
        
        self.tau = tau
        self.gamma = gamma
        self.entropy_bound = entropy_bound
    
    def take_action(self, s):
        s = torch.tensor(s.reshape(1, -1), dtype=torch.float)
        a = self.actor(s)[0].item()
        return [a]
    
    def update(self, transition):
        s = torch.tensor(transition['state'], dtype=torch.float)
        s_ = torch.tensor(transition['next_state'], dtype=torch.float)
        a = torch.tensor(transition['action'], dtype=torch.float)
        r = torch.tensor(transition['reward'], dtype=torch.float)
        d = torch.tensor(transition['done'], dtype=torch.int64)

        r = (r + 8.) / 8.
        a_, log_proba = self.actor(s_)
        td_target = r + self.gamma * (1 - d) * \
            (torch.min(self.critic_target1(s_, a_), self.critic_target2(s_, a_)) - torch.exp(self.alpha) * log_proba)
        loss_critic1 = F.mse_loss(self.critic1(s, a), td_target.detach())
        self.critic_optimizer1.zero_grad()
        loss_critic1.backward()
        self.critic_optimizer1.step()
        loss_critic2 = F.mse_loss(self.critic2(s, a), td_target.detach())
        self.critic_optimizer2.zero_grad()
        loss_critic2.backward()
        self.critic_optimizer2.step()

        a, log_proba = self.actor(s)
        loss_actor = torch.mean(torch.exp(self.alpha) * log_proba - torch.min(self.critic1(s, a), self.critic2(s, a)))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        loss_alpha = torch.mean(-torch.exp(self.alpha) * (log_proba + self.entropy_bound).detach())
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()

        for p_, p in zip(self.critic_target1.parameters(), self.critic1.parameters()):
            p_.data.copy_((1 - self.tau) * p_.data + self.tau * p.data)
        for p_, p in zip(self.critic_target2.parameters(), self.critic2.parameters()):
            p_.data.copy_((1 - self.tau) * p_.data + self.tau * p.data)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, hidden_dim, action_dim, gamma=0.99, tau=0.005, entropy_bound=-env.action_space.shape[0], 
                 lr_critic=3e-3, lr_actor=3e-4, lr_alpha=3e-4, action_bound=env.action_space.high[0])
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    eposides = 100
    returns = np.zeros(eposides)
    replaybuffer = ReplayBuffer(max_len=100000)
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