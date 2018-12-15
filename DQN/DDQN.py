import tensorflow as tf
import numpy as np
from dqn import deep_q_net

np.random.seed(1)
tf.set_random_seed(1)

class double_dqn(deep_q_net):

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

        # 这一段和 DQN 不一样
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.features:],  # next observation
                       self.s: batch_memory[:, -self.features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.features]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.bs, dtype=np.int32)
        eval_act_index = batch_memory[:, self.features].astype(int)
        reward = batch_memory[:, self.features + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)   # q_eval下得分最高的动作
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN 选择 q_next 依据 q_eval 选出的动作

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # 训练 eval_net
        _, self.cost = self.sess.run([self.train_op, self.loss], feed_dict={
            self.s: batch_memory[:, :self.features], self.q_target: q_target
        })
        self.cost_history.append(self.cost)
        # 逐渐增加 epsilon, 降低行为的随机性
        self.epi = self.epi + self.epi_inc if self.epi < self.epi_max else self.epi_max
        self.learn_step_counter += 1