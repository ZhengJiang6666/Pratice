import gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim),
                                 nn.Softmax(dim=-1),
                                 )
        
    def forward(self, x):
        return self.net(x)

class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma):
        self.net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
    
    def take_aciton(self, s):
        s = torch.tensor(s.reshape(1, -1), dtype=torch.float)
        a = self.net(s)
        m = torch.distributions.Categorical(probs=a)
        return m.sample().item()
    
    def update(self, transition):
        s = torch.tensor(transition['state'], dtype=torch.float)
        a = torch.tensor(transition['action'], dtype=torch.int64)
        r = torch.tensor(transition['reward'], dtype=torch.float)

        g = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(r))):
            g = self.gamma * g + r[i].item()
            loss = -torch.log(self.net(s[[i], :]).gather(1, a[[i], :])) * g
            loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    lr = 1e-3
    gamma = 0.98
    episodes = 1000
    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr=lr, gamma=gamma)
    returns = np.zeros(episodes)

    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    
    for i in tqdm(range(episodes)):
        state = env.reset()
        done = False
        actions, rewards, states = [], [], []

        while not done:
            action = agent.take_aciton(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            returns[i] += reward
        
        transition = {'state': np.array(states), 
                      'action': np.array(actions).reshape(-1, 1),
                      'reward': np.array(rewards).reshape(-1, 1)}
        agent.update(transition)
    
    plt.figure()
    plt.plot(range(episodes), returns)
    plt.show()