import numpy as np
import time
from env import Env

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 30
LAMBDA=0.9
EPOCH=100

np.random.seed(1)
#### epsilon贪心算法
def epsilon_greedy(Q, state):
    n=np.random.uniform()

    if (n > 1 - EPSILON) or ((Q[state, :] == 0).all()):
        action = np.random.randint(0, 4)  # 0~3
    else:
        action = Q[state, :].argmax()
    return action

e = Env()
#### Q表：Q(s,a)
Q = np.zeros((e.state_num, 4))

for i in range(EPOCH):
    e = Env()
    #### E表：E(s,a)
    ### 1. 主要记录所经历过的路径
    ### 2. 即最终产生的reward变化跟所经历过的所有路径有关
    E = np.zeros((e.state_num, 4))
    #### e.is_end !=False
    while ((e.is_end==0) and (e.step < MAX_STEP)):
        action = epsilon_greedy(Q, e.present_state)
        state = e.present_state
        reward = e.interact(action)
        new_state = e.present_state
        new_action= epsilon_greedy(Q,e.present_state)
        E[state,action]+=1
        delta = reward+GAMMA*Q[new_state,new_action]-Q[state,action]
        for s in range(e.state_num):
            for a in range(4):
                Q[s,a]+=ALPHA*delta*E[s,a]
                E[s,a]=E[s,a]*LAMBDA*GAMMA
        e.print_map()
    print('Episode:', i, 'Total Step:', e.step, 'Total Reward:', e.total_reward)
