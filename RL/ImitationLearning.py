import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PPO import PPO, Actor

def pretrain(env):
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, lr_actor=1e-3, lr_critic=1e-2,
                gamma=0.98, lambd=0.95, epislon=0.2, epoch=10)
    eposides = 250
    returns = np.zeros(eposides)

    for i in tqdm(range(eposides)):
        state = env.reset()
        done = False
        states, actions, rewards, dones, next_states = [], [], [], [], []

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            actions.append(action)
            dones.append(done)
            state = next_state
            returns[i] += reward
        
        transition = {'state': np.array(states),
                      'next_state': np.array(next_states),
                      'reward': np.array(rewards).reshape(-1, 1),
                      'action': np.array(actions).reshape(-1, 1),
                      'done': np.array(dones).reshape(-1, 1)}
        agent.update(transition)
    
    # plt.figure()
    # plt.plot(range(eposides), returns)
    # plt.show()
    
    return agent

def sample_sa(agent, env, n_eposide):
    states, actions = [], []
    for i in range(n_eposide):
        s = env.reset()
        d = False
        while not d:
            a = agent.take_action(s)
            s_, r, d, _ = env.step(a)
            states.append(s)
            actions.append(a)
            s = s_
    return np.array(states), np.array(actions).reshape(-1, 1)

class BC:
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr)
    
    def take_action(self, s):
        s = torch.tensor(s.reshape(1, -1), dtype=torch.float)
        proba = self.actor(s)
        dist = torch.distributions.Categorical(probs=proba)
        a = dist.sample().item()
        return a
    
    def learn(self, es, ea):
        es = torch.tensor(es, dtype=torch.float)
        ea = torch.tensor(ea, dtype=torch.int64)
        pred_proba = torch.log(self.actor(es).gather(1, ea))
        loss = -pred_proba.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def test_agent(env, agent, n_eposide):
    return_list = []
    for _ in range(n_eposide):
        state = env.reset()
        done = False
        return_value = 0
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            return_value += reward
            state = next_state
        return_list.append(return_value)
    return np.mean(return_list)

def mainBC():
    env = gym.make('CartPole-v0')
    env.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ppo_agent = pretrain(env)
    states, actions = sample_sa(ppo_agent, env, n_eposide=1)
    idx = random.sample(range(states.shape[0]), k=30)
    es = states[idx, :]
    ea = actions[idx, :]

    n_iterations = 1000
    batch_size = 64
    agent = BC(state_dim=env.observation_space.shape[0], hidden_dim=128, action_dim=env.action_space.n, lr=1e-3)
    returns = np.zeros(n_iterations)
    
    for i in tqdm(range(n_iterations)):
        idx = np.random.randint(low=0, high=es.shape[0], size=batch_size)
        agent.learn(es[idx, :], ea[idx, :])
        returns[i] = test_agent(env, agent, n_eposide=5)
    
    plt.figure()
    plt.plot(range(n_iterations), returns)
    plt.show()

class Discirminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1),
                                 nn.Sigmoid())
    
    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = self.net(x)
        return x

class GAIL:
    def __init__(self, state_dim, hidden_dim, action_dim, lr_d):
        self.discriminator = Discirminator(state_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr_d)
        self.agent = PPO(state_dim, hidden_dim, action_dim, lr_actor=1e-3, lr_critic=1e-2,
                         gamma=0.98, lambd=0.95, epislon=0.2, epoch=10)
    
    def take_action(self, s):
        return self.agent.take_action(s)
    
    def learn(self, s, a, es, ea, s_, d):
        fs = torch.tensor(s, dtype=torch.float)
        fa = torch.tensor(a, dtype=torch.int64)
        es = torch.tensor(es, dtype=torch.float)
        ea = torch.tensor(ea, dtype=torch.int64)
        fa = F.one_hot(fa.squeeze(), 2).float()
        ea = F.one_hot(ea.squeeze(), 2).float()
        
        p_f = self.discriminator(fs, fa)
        p_e = self.discriminator(es, ea)
        loss = F.mse_loss(p_f, torch.ones_like(p_f)) + F.mse_loss(p_e, torch.zeros_like(p_e))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        r = -p_f.log().detach()
        transition = {'state': s, 'action': a, 'next_state': s_, 'done': d, 'reward': r}
        self.agent.update(transition)

def mainGAIL():
    env = gym.make('CartPole-v0')
    env.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ppo_agent = pretrain(env)
    states, actions = sample_sa(ppo_agent, env, n_eposide=1)
    idx = random.sample(range(states.shape[0]), k=30)
    es = states[idx, :]
    ea = actions[idx, :]

    n_eposides = 500
    agent = GAIL(state_dim=env.observation_space.shape[0], hidden_dim=128, action_dim=env.action_space.n, 
                 lr_d=1e-3)
    returns = np.zeros(n_eposides)
    
    for i in tqdm(range(n_eposides)):
        state = env.reset()
        done = False
        states, next_states, actions, dones = [], [], [], []
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            next_states.append(next_state)
            dones.append(done)
            actions.append(action)
            returns[i] += reward
            state = next_state
        agent.learn(np.array(states), np.array(actions).reshape(-1, 1), 
                    es, ea, np.array(next_states), np.array(dones).reshape(-1, 1))
    
    plt.figure()
    plt.plot(range(n_eposides), returns)
    plt.show()

if __name__ == '__main__':
    # mainBC()
    mainGAIL()