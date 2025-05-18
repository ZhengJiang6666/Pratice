import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim),
                                 nn.Softmax(dim=-1))
    
    def forward(self, x):
        return self.net(x)

class ActorContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.linear = nn.Linear(state_dim, hidden_dim)
        self.ave = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear(x))
        ave = self.tanh(self.ave(x)) * 2 #pendulum action space -2 to 2
        std = self.softplus(self.std(x))
        return ave, std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, value_dim=1):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, value_dim))
    
    def forward(self, x):
        return self.net(x)

class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, 
                 lr_actor, lr_critic, gamma, lambd, epislon, epoch):
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(state_dim, hidden_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.lambd = lambd
        self.epislon = epislon
        self.epoch = epoch
    
    def take_action(self, s):
        s = torch.tensor(s.reshape(1, -1), dtype=torch.float)
        a = self.actor(s)
        m = torch.distributions.Categorical(probs=a)
        return m.sample().item()
    
    def update(self, transition):
        s = torch.tensor(transition['state'], dtype=torch.float)
        a = torch.tensor(transition['action'], dtype=torch.int64)
        r = torch.tensor(transition['reward'], dtype=torch.float)
        s_ = torch.tensor(transition['next_state'], dtype=torch.float)
        d = torch.tensor(transition['done'], dtype=torch.int64)

        td_target = r + self.gamma * self.critic(s_) * (1 - d)
        td_delta = td_target - self.critic(s)
        advantage = self.GAE(td_delta.detach())
        policy = self.actor(s).gather(1, a)

        for _ in range(self.epoch):
            impt = self.importance(self.actor(s).gather(1, a), policy.detach())
            cliped = torch.clip(impt, 1-self.epislon, 1+self.epislon) * advantage
            loss_actor = torch.mean(-torch.min(impt * advantage, cliped))
            loss_critic = F.mse_loss(self.critic(s), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss_actor.backward()
            loss_critic.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def GAE(self, td_delta):
        advantage = np.zeros(td_delta.shape)
        a = 0
        for i in reversed(range(len(td_delta))):
            a = self.gamma * self.lambd * a + td_delta[i].item()
            advantage[i, :] = a
        advantage = torch.tensor(advantage, dtype=torch.float)
        return advantage
    
    def importance(self, p_, p):
        m = torch.log(p_) - torch.log(p)
        m = torch.exp(m)
        return m

class PPOContinuous(PPO):
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, lambd, epislon, lr_actor, lr_critic, epoch):
        self.actor = ActorContinuous(state_dim, hidden_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = Critic(state_dim, hidden_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.lambd = lambd
        self.epislon = epislon
        self.epoch = epoch
    
    def take_action(self, s):
        s = torch.tensor(s.reshape(1, -1), dtype=torch.float)
        ave, std = self.actor(s)
        dist = torch.distributions.Normal(ave, std)
        return [dist.sample().item()]
    
    def update(self, transition):
        s = torch.tensor(transition['state'], dtype=torch.float)
        s_ = torch.tensor(transition['next_state'], dtype=torch.float)
        a = torch.tensor(transition['action'], dtype=torch.float)
        r = torch.tensor(transition['reward'], dtype=torch.float)
        d = torch.tensor(transition['done'], dtype=torch.int64)

        td_target = (r + 8) / 8 + self.gamma * self.critic(s_) * (1 - d)
        td_delta = td_target - self.critic(s)
        advantage = self.GAE(td_delta.detach())
        mu, sigma = self.actor(s)
        p = torch.distributions.Normal(mu.detach(), sigma.detach()).log_prob(a)
        
        for _ in range(self.epoch):
            mu, sigma = self.actor(s)
            p_ = torch.distributions.Normal(mu, sigma).log_prob(a)
            impt = torch.exp(p_ - p)
            actor_loss = torch.mean(-torch.min(impt * advantage, 
                                               torch.clip(impt, 1-self.epislon, 1+self.epislon) * advantage))
            critic_loss = F.mse_loss(self.critic(s), td_target.detach())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    env.seed(0)
    state_dim = env.observation_space.shape[0]
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
    agent = PPOContinuous(state_dim, hidden_dim, action_dim, lr_actor=1e-4, lr_critic=5e-3,
                gamma=0.9, lambd=0.9, epislon=0.2, epoch=10)
    eposides = 2000
    returns = np.zeros(eposides)
    torch.manual_seed(0)

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
    
    plt.figure()
    plt.plot(range(eposides), returns)
    plt.show()