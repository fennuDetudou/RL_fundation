import gym
import argparse
from dqn import deep_q_net
from DDQN import double_dqn
from duelingdqn import dueling_dqn

parser=argparse.ArgumentParser()
parser.add_argument('--model',default='dqn',help='input the method')

env = gym.make('CartPole-v0')   # 定义使用gym库中的那一个环境
env = env.unwrapped # 不做这个会有很多限制

args=parser.parse_args()

if args.model=='dqn':
    RL = deep_q_net(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      learning_rate=0.01, e_greedy=0.9,
                      replace_target_iter=100, memory_size=2000,
                      e_greedy_increment=0.0008,output_graph=True)
elif args.model=='ddqn':
    RL = double_dqn(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.01, e_greedy=0.9,
                    replace_target_iter=100, memory_size=2000,
                    e_greedy_increment=0.0008, output_graph=True)

elif args.model=='dueling':
    RL = dueling_dqn(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.01, e_greedy=0.9,
                    replace_target_iter=100, memory_size=2000,
                    e_greedy_increment=0.0008, output_graph=True)
else:
    raise TypeError("wrong model!")


total_steps = 0 # 记录步数

for i_episode in range(100):

    # 获取回合 i_episode 第一个 observation
    observation = env.reset()
    ep_r = 0
    while True:
        env.render()    # 刷新环境

        action = RL.choose_action(observation)  # 选行为

        observation_,_, done, info = env.step(action) # 获取下一个 state

        x, x_dot, theta, theta_dot = observation_   # 细分开, 为了修改原配的 reward

        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高

        #x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2   # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率

        # 保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()  # 学习

        ep_r += reward
        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epi, 2))
            break

        observation = observation_
        total_steps += 1
# 最后输出 cost 曲线
RL.plot_cost()