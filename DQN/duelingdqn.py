import tensorflow as tf
import numpy as np
from dqn import deep_q_net
np.random.seed(1)
tf.set_random_seed(1)

class dueling_dqn(deep_q_net):

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('conv1'):
                w1 = tf.get_variable('w1', [self.features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2', [n_l1, 1], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, 1], initializer=b_initializer,
                                     collections=c_names)
                self.V = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2', [n_l1, self.actions], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.actions], initializer=b_initializer,
                                     collections=c_names)
                self.A = tf.matmul(l1, w2) + b2

            with tf.variable_scope('Q'):
                # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)

            return out

        # -------------- 创建 eval 神经网络, 及时提升参数 --------------
        self.s = tf.placeholder(tf.float32, [None, self.features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.actions], name='q_target')
        with tf.variable_scope('eval_net'):
            ##### 获取参数必要的步骤， c_names(collection_names)更新参数时用到
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.truncated_normal_initializer(), tf.constant_initializer(0.1)

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.q_target))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------------- 创建 target 神经网络, 提供 target Q ---------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.features], name='s_')  # 接收下个 observation
        with tf.variable_scope('target_net'):
            # c_names(collections_names) 是在更新 target_net 参数时会用到
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)