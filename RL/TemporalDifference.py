import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functools import reduce

class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x

# =============================================================================
# 
# =============================================================================
class TemporalDifference:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_step=None, n_action=4):
        self.q = np.zeros((ncol*nrow, n_action), dtype=float)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        self.n_step = n_step
        self.states = []
        self.rewards = []
        self.actions = []
    
    def take_action(self, s):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.q[s]),)
        else:
            return np.argmax(self.q[s])
    
    def best_action(self, s):
        a = np.zeros(4)
        a[np.where(self.q[s] == max(self.q[s]))] = 1
        return a
    
    def sarsa(self, s, a, r, s_, a_):
        error = self.q[s, a] - r - self.gamma * self.q[s_, a_]
        self.q[s, a] -= self.alpha * error
    
    def nstep_sarsa(self, s, a, r, s_, a_, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        if len(self.states) == self.n_step:
            s = self.states.pop(0)
            a = self.actions.pop(0)
            self.q[s, a] -= self.alpha * (self.q[s, a] - 
                                          reduce(lambda x, y : self.gamma * x + y, 
                                                  reversed(self.rewards + [self.q[s_, a_]]))
                                          )
        # if len(self.states) == self.n_step:  # 若保存的数据可以进行n步更新
        #     G = self.q[s, a]  # 得到Q(s_{t+n}, a_{t+n})
        #     for i in reversed(range(self.n_step)):
        #         G = self.gamma * G + self.rewards[i]  # 不断向前计算每一步的回报
        #         # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
        #         if done and i > 0:
        #             s = self.states[i]
        #             a = self.actions[i]
        #             self.q[s, a] += self.alpha * (G - self.q[s, a])
        #     s = self.states.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
        #     a = self.actions.pop(0)
        #     self.rewards.pop(0)
        #     # n步Sarsa的主要更新步骤
        #     self.q[s, a] += self.alpha * (G - self.q[s, a])
            
        if done:
            self.states = []
            self.actions = []
            self.rewards = []
    
    def q_learning(self, s, a, r, s_):
        self.q[s, a] -= self.alpha * (self.q[s, a] - r - self.gamma * max(self.q[s_]))

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

if __name__ == '__main__':
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = TemporalDifference(ncol, nrow, epsilon, alpha, gamma, n_step=4)
    num_episodes = 500
    
    returns = []
    for i in range(num_episodes):
        state = env.reset()
        action = agent.take_action(state)
        done = False
        return_value = 0
        
        while not done:
            next_state, reward, done= env.step(action)
            next_action = agent.take_action(next_state)
            agent.nstep_sarsa(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            return_value += reward
        
        returns.append(return_value)
    
    plt.figure()
    plt.plot(range(len(returns)), returns)
    plt.show()
    
    action_meaning = ['^', 'v', '<', '>']
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])