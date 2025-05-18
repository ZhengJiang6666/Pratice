import gym
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ValueNet(nn.ModuleList):
    def __init__(self, state_dim, hidden_dim, value_dim=1):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, value_dim))
        
    def forward(self, x):
        return self.net(x)

class PolicyNet(nn.ModuleList):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim),
                                 nn.Softmax(dim=-1))
    
    def forward(self, x):
        return self.net(x)

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_value, lr_policy, gamma):
        self.value_net = ValueNet(state_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.gamma = gamma

    def take_action(self, s):
        s = torch.tensor(s.reshape(1, -1), dtype=torch.float)
        a = self.policy_net(s)
        m = torch.distributions.Categorical(probs=a)
        return m.sample().item()
    
    def update(self, transition):
        s = torch.tensor(transition['state'], dtype=torch.float)
        a = torch.tensor(transition['action'], dtype=torch.int64)
        s_ = torch.tensor(transition['next_state'], dtype=torch.float)
        r = torch.tensor(transition['reward'], dtype=torch.float)
        d = torch.tensor(transition['done'], dtype=torch.int64)

        vs = self.value_net(s)
        vs_ = self.value_net(s_)
        target = r + self.gamma * vs_ * (1 - d)
        delta =  target - vs

        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        value_loss = F.mse_loss(vs, target.detach())
        policy_loss = torch.mean(-torch.log(self.policy_net(s).gather(1, a)) * delta.detach())
        value_loss.backward()
        policy_loss.backward()
        self.value_optimizer.step()
        self.policy_optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    env.seed(0)
    agent = ActorCritic(state_dim, hidden_dim, action_dim, lr_policy=1e-3, lr_value=1e-2, gamma=0.98)
    np.random.seed(0)
    torch.manual_seed(0)
    eposides = 1000
    returns = np.zeros(eposides)

    for i in tqdm(range(eposides)):
        state = env.reset()
        done = False
        states, next_states, rewards, actions, dones = [], [], [], [], []

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            returns[i] += reward
        
        transition = {'state': np.array(states),
                      'next_state': np.array(next_states),
                      'action': np.array(actions).reshape(-1, 1),
                      'reward': np.array(rewards).reshape(-1, 1),
                      'done': np.array(dones).reshape(-1, 1)}
        agent.update(transition)
    
    plt.figure()
    plt.plot(range(eposides), returns)
    plt.show()