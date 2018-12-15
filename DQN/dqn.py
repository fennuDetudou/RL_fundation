import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

class deep_q_net(object):
    '''
    f(s)=Q(s,a1,w),Q(s,a2,w),...
    '''

    #### 初始化
    def __init__(self,n_actions,n_features,
                 learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,
                 replace_target_iter=200,memory_size=500,batch_size=32,
                 e_greedy_increment=None,output_graph=True):

        ### 与环境相关的状态空间和动作空间定义
        self.actions=n_actions
        self.features=n_features
        ### q_net参数定义
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epi_max=e_greedy
        self.replace_target=replace_target_iter  # 更换target—network的步数
        self.experice_pool=memory_size   # 经验池
        self.bs=batch_size
        self.epi_inc=e_greedy_increment   # epi的增量
        self.epi=0 if e_greedy_increment is not None else self.epi_max #是否开启探索模式
        # 记录学习的步骤，判断是否需要更换target—network
        self.learn_step_counter=0
        # 初始化经验池，（s,a,r,s_)
        self.experice=np.zeros((self.experice_pool,n_features*2+2))
        self.cost_history=[]   # 损失函数历史记录

        # target_net 和 eval_net
        self._build_net()
        ####  替换target 参数
        ### get_collection返回的是参数列表
        t_params=tf.get_collection('target_net_params')
        e_params=tf.get_collection('eval_net_params')
        self.replace_target_op=[tf.assign(t,e) for (t,e) in zip(t_params,e_params)]

        self.sess=tf.Session()
        if output_graph:
            tf.summary.FileWriter('logs/',self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    #### 建立Q_NET
    def _build_net(self):

        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        self.s=tf.placeholder(tf.float32,[None,self.features],name='s')
        self.q_target=tf.placeholder(tf.float32,[None,self.actions],name='q_target')
        with tf.variable_scope('eval_net'):
            ##### 获取参数必要的步骤， c_names(collection_names)更新参数时用到
            c_names,n_l1,w_initializer,b_initializer=\
                ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES],10,\
                tf.truncated_normal_initializer(),tf.constant_initializer(0.1)

            with tf.variable_scope('conv1'):
                w1=tf.get_variable('w1',[self.features,n_l1],tf.float32,w_initializer,
                                   collections=c_names)
                b1=tf.get_variable('b1',[1,n_l1],tf.float32,b_initializer,
                                   collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s,w1)+b1)

            with tf.variable_scope('f_c'):
                w2 = tf.get_variable('w2', [n_l1, self.actions], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=b_initializer,
                                     collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
                self.loss=tf.reduce_mean(tf.squared_difference(self.q_eval,self.q_target))

        with tf.variable_scope('train'):
                self.train_op=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.features], name='s_')  # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # target_net 的第一层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('conv1'):
                w1 = tf.get_variable('w1', [self.features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1,n_l1], initializer=b_initializer,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # target_net 的第二层. collections 是在更新 target_net 参数时会用到
            with tf.variable_scope('fc'):
                w2 = tf.get_variable('w2', [n_l1, self.actions], initializer=w_initializer,
                                     collections=c_names)
                b2=tf.get_variable('b2',[1,self.actions],initializer=b_initializer,
                                   collections=c_names)
                self.q_next=tf.matmul(l1,w2)+b2

    #### 经验池
    def store_transition(self,s,a,r,s_):

        #### 经验池计数器
        if not hasattr(self,'experice_counter'):
            self.experice_counter=0
        transition=np.hstack((s,[a,r,],s_))

        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.experice_counter % self.experice_pool
        self.experice[index, :] = transition  # 替换过程
        self.experice_counter += 1

    #### 行为选择
    def choose_action(self,observation):
        # 统一 observation 的 shape (1, size_of_observation)
        observation=observation[np.newaxis,:]

        if np.random.uniform()<self.epi:   #  选择最优的动作
            action_value=self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action=np.argmax(action_value)

        else:   # 探索
            action=np.random.randint(0,self.actions)

        return action

    def learn(self):

        # 检查是否替换 target_net 参数
        if self.learn_step_counter % self.replace_target == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从经验池中选取batch_size 个 e
        if self.experice_pool > self.experice_counter:
            sample_index = np.random.choice(self.experice_counter, size=self.bs)
        else:
            sample_index = np.random.choice(self.experice_pool, size=self.bs)
        batch_memory = self.experice[sample_index, :]

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                # batch_memory中特征为[s,a,r,s_]
                self.s_: batch_memory[:, -self.features:],
                self.s: batch_memory[:, :self.features]
            })

        # 下面这几步十分重要. q_next, q_eval 包含所有 action 的值,
        # 而我们需要的只是已经选择好的 action 的值, 其他的并不需要.
        # 所以我们将其他的 action 值全变成 0, 将用到的 action 误差值 反向传递回去, 作为更新凭据.
        # 这是我们最终要达到的样子, 比如 q_target - q_eval = [1, 0, 0] - [-1, 0, 0] = [2, 0, 0]
        # q_eval = [-1, 0, 0] 表示这一个记忆中有我选用过 action 0, 而 action 0 带来的 Q(s, a0) = -1,
# 所以其他的 Q(s, a1) = Q(s, a2) = 0.
        # q_target = [1, 0, 0] 表示这个记忆中的 r+gamma*maxQ(s_) = 1, 而且不管在 s_ 上我们取了哪个 action,
        # 我们都需要对应上 q_eval 中的 action 位置, 所以就将 1 放在了 action 0 的位置.

        # 下面也是为了达到上面说的目的, 不过为了更方面让程序运算, 达到目的的过程有点不同.
        # 是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,
        # 不过 我们再根据 batch_memory 当中的 action 这个 column 来给 q_target 中的对应的 memory-action 位置来修改赋值.
        # 使新的赋值为 reward + gamma * maxQ(s_), 这样 q_target-q_eval 就可以变成我们所需的样子.

        #### 即要更新的参数只有一个动作，虽然网络的输出包括所有动作，但是由于r只跟当前的a有关，所以我们只跟新当前的a，
    #### 所以难点在于怎么将他们有数值的那部分对应起来
        q_target = q_eval.copy()
        batch_index = np.arange(self.bs, dtype=np.int32)
        eval_act_index = batch_memory[:, self.features].astype(int) # 采取的动作的索引
        reward = batch_memory[:, self.features + 1]
        #### 本来是将 q_eval 全部赋值给 q_target, 这时 q_target-q_eval 全为 0,但是我们改变相应的索引的q_target的值
        #### 这样就达到了我们更新的目的
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        #训练 eval_net
        _,self.cost=self.sess.run([self.train_op,self.loss],feed_dict={
            self.s:batch_memory[:,:self.features],self.q_target:q_target
        })
        self.cost_history.append(self.cost)
        # 逐渐增加 epsilon, 降低行为的随机性
        self.epi = self.epi + self.epi_inc if self.epi < self.epi_max else self.epi_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

