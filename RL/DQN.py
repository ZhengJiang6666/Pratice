import collections
import torch.nn as nn
import torch
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# 
# =============================================================================
class ReplayBuffer:
    def __init__(self, experiences):
        self.buffer = collections.deque(maxlen=experiences)
    
    def sample(self, batch_size):
        s, a, r, s_, d = zip(*random.sample(self.buffer, batch_size))
        s = torch.tensor(np.array(s), dtype=torch.float)
        a = torch.tensor(a, dtype=torch.int64).reshape(-1, 1)
        r = torch.tensor(r, dtype=torch.float).reshape(-1, 1)
        s_ = torch.tensor(np.array(s_), dtype=torch.float)
        d = torch.tensor(d, dtype=torch.int64).reshape(-1, 1)
        return s, a, r, s_, d
    
    def add(self, item):
        self.buffer.append(item)

    @property
    def size(self):
        return len(self.buffer)
    
class Qnet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim)
                                 )
    
    def forward(self, x):
        return self.mlp(x)

class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, epsilon, lr, n):
        self.q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.q_net_delay = Qnet(state_dim, hidden_dim, action_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.step = 0
        self.action_dim = action_dim

    def update(self, s, a, r, s_, done):
        qk = self.q_net(s)
        qk = torch.gather(qk, dim=1, index=a)
        qk1 = self.q_net_delay(s_)
        qk1 = torch.max(qk1, dim=1, keepdim=True)[0]
        loss = self.criterion(qk, r + self.gamma * qk1 *(1 - done))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step += 1
        if self.step % self.n == 0:
            self.q_net_delay.load_state_dict(self.q_net.state_dict())
    
    def take_action(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            s = torch.tensor(s.reshape(1,-1), dtype=torch.float)
            return torch.argmax(self.q_net(s), dim=1).item()

if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    replaybuffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, gamma, epsilon, lr, n=target_update)

    returns = np.zeros(num_episodes)
    for i in tqdm(range(num_episodes)):
        # print(i, end=' ', flush=True)
        state = env.reset()
        done = False
        
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            replaybuffer.add((state, action, reward, next_state, done))
            state = next_state
            returns[i] += reward

            if replaybuffer.size < minimal_size:
                continue
            batch_data = replaybuffer.sample(batch_size)
            agent.update(*batch_data)
    
    plt.figure()
    plt.plot(range(num_episodes), returns)
    plt.show()
