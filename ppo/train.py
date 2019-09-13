from ppo.config import config
from ppo.agent import PPOAgent
from ppo.env import *
import tensorflow as tf
from ppo.policy_networks import *


def train():
    env = SketchDesigner(SketchDiscriminator('./saved_gan/1'))
    policy = Policy('policy')
    old_policy = Policy('old_policy')
    sess = tf.Session()

    ppo = PPOAgent(sess=sess, policy=policy, old_policy=old_policy)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./log/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    state, reward, _ = env.new_canvas()
    state = state.reshape([-1] + config['STATE_DIM'] + [1])

    #TODO: finish train loop
    for iteration in range(config['MAX_ITERATION']):
        states = []
        actions = []
        v_preds = []
        rewards = []



        for steps in range(100):
            act, v_pred = ppo.act(state, stochastic=True)







