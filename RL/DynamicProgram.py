import copy
import numpy as np

class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol * self.env.nrow)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  ##rewrite
        while True: #迭代求解state value
            max_diff = 0
            cal_qsa = lambda x : sum([y[0] * y[2] for y in x]) + sum([y[0] * self.v[y[1]] * self.gamma for y in x])
            
            for s in range(self.env.nrow * self.env.ncol):
                qsa = [cal_qsa(self.env.P[s][a]) for a in range(4)]
                vs = sum(np.array(self.pi[s]) * np.array(qsa))
                max_diff = max(max_diff, abs(vs - self.v[s]))
                self.v[s] = vs
            
            if max_diff < self.theta:
                break

    def policy_improvement(self):  ##rewrite
        cal_qsa = lambda x : sum([y[0] * y[2] for y in x]) + sum([y[0] * self.v[y[1]] * self.gamma for y in x])
        
        for s in range(self.env.nrow * self.env.ncol):
            qsa = [cal_qsa(self.env.P[s][a]) for a in range(4)]
            qsa = np.array(qsa)
            idx = np.where(qsa == np.max(qsa))[0] #better action state used to update pi
            self.pi[s] = [1/len(idx) if i in idx else 0 for i in range(4)] #update pi
        
        return self.pi

    def policy_iteration(self):  ##rewrite
        while True:
            self.policy_evaluation()
            pi = copy.deepcopy(self.pi)
            self.policy_improvement()
            if pi == self.pi:
                break

class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self): #rewrite
        while True:
            max_diff = 0
            cal_qsa = lambda x : sum([y[0] * y[2] for y in x]) + sum([y[0] * self.v[y[1]] * self.gamma for y in x])
            
            for s in range(self.env.nrow * self.env.ncol):
                qsa = [cal_qsa(self.env.P[s][a]) for a in range(4)]
                max_diff = max(max_diff, abs(max(qsa) - self.v[s]))
                self.v[s] = max(qsa)
            
            if max_diff < self.theta:
                break
        
        self.get_policy()

    def get_policy(self):  #rewrite
        cal_qsa = lambda x : sum([y[0] * y[2] for y in x]) + sum([y[0] * self.v[y[1]] * self.gamma for y in x])
        for s in range(self.env.nrow * self.env.ncol):
            qsa = [cal_qsa(self.env.P[s][a]) for a in range(4)]
            qsa = np.array(qsa)
            idx = np.where(qsa == max(qsa))[0]
            self.pi[s] = [1/len(idx) if i in idx else 0 for i in range(4)]

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = ValueIteration(env, theta, gamma)
# agent.value_iteration()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])

import gym

env = gym.make('FrozenLake-v0')
env = env.unwrapped
env.render()
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5,7,11,12], [15])
