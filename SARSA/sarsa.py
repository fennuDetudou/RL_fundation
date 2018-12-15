import numpy as np
import time
from env import Env

EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_STEP = 30
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
#### Q表:Q(s,a)
Q = np.zeros((e.state_num, 4))

for i in range(EPOCH):
    e = Env()
    #### e.is_end !=False
    while ((e.is_end==0) and (e.step < MAX_STEP)):
        action = epsilon_greedy(Q, e.present_state)
        state = e.present_state
        reward = e.interact(action)
        new_state = e.present_state
        new_action= epsilon_greedy(Q,e.present_state)
        #这样的话更新比较平滑，防止模型过旱收敛到局部极小值。
        Q[state, action] = (1 - ALPHA) * Q[state, action] + \
            ALPHA * (reward + GAMMA * Q[new_state,new_action])

        e.print_map()
        #### 可视化学习过程
        #time.sleep(0.1)
    print('Episode:', i, 'Total Step:', e.step, 'Total Reward:', e.total_reward)

