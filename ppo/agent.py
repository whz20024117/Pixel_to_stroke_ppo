import tensorflow as tf
from copy import deepcopy
from ppo.config import config
from ppo.distribution import DiagGaussianPdType


class PPO:
    def __init__(self, sess, policy, old_policy, gamma=config['GAMMA'], clip_value=config['CLIP']):
        self.sess = sess
        self.policy = policy
        self.old_policy = old_policy
        self.gamma = gamma

        policy_trainable = self.policy.get_trainable_variables()
        old_policy_trainable = self.old_policy.get_trainable_variables()

        with tf.variable_scope('assign_ops'):
            self.assign_ops = []
            for v_old, v in zip(old_policy_trainable, policy_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        with tf.variable_scope('train_inputs'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.advantages = tf.placeholder(dtype=tf.float32, shape=[None], name='advantages')

        action_vec = self.policy.action_pred
        action_vec_old = self.old_policy.action_pred

        pdType = DiagGaussianPdType(size=config['ACTION_DIM'])
        self.pd, self.pi_mean = pdType.pdfromlatent(action_vec)
        self.pd_old, self.pi_mean_old = pdType.pdfromlatent(action_vec_old)

        with tf.variable_scope('loss/clip'):

            ratio = tf.exp(self.pd_old.neglogp(action_vec_old) - self.pd.neglogp(action_vec))
            clipped_ratios = tf.clip_by_value(ratio, clip_value_min=1-config['CLIP'], clip_value_max=1+config['CLIP'])
            loss_clip = tf.minimum(tf.multiply(self.advantages, ratio), tf.multiply(self.advantages, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)

            tf.summary.scalar('loss_clip', loss_clip)

        with tf.variable_scope('loss/value_function'):
            v_preds = self.policy.value_pred
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)

            tf.summary.scalar('loss_value_function', loss_vf)

        with tf.variable_scope('loss/entropy'):
            entropy = tf.reduce_mean(self.pd.entropy())

            tf.summary.scalar('entropy', entropy)

        with tf.variable_scope('loss'):
            loss = loss_clip - config['C1'] * loss_vf + config['C2'] * entropy
            loss = -loss
            tf.summary.scalar('loss', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(config['LR'])
        self.train_op = optimizer.minimize(loss, var_list=self.policy.get_trainable_variables())

#TODO: train method



    def get_summary(self, state, actions, rewards, v_preds_next, advantages):
        return self.sess.run(self.merged, feed_dict={self.policy.state: state,
                                                     self.old_policy.state: state,
                                                     self.actions: actions,
                                                     self.rewards: rewards,
                                                     self.v_preds_next: v_preds_next,
                                                     self.advantages: advantages})

    def assign_parameters(self):
        return self.sess.run(self.assign_ops)

    def get_advantages(self, rewards, v_preds, v_preds_next):
        # https://github.com/uidilr/ppo_tf/blob/master/ppo.py

        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

