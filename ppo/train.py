from ppo.config import config
from ppo.agent import PPOAgent
from ppo.env import *
import tensorflow as tf
from ppo.policy_networks import *


def train(sess):
    env = SketchDesigner(SketchDiscriminator(config['SAVED_GAN']))
    policy = Policy('policy')
    old_policy = Policy('old_policy')

    ppo = PPOAgent(sess=sess, policy=policy, old_policy=old_policy)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./log/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    for iteration in range(config['MAX_ITERATION']):
        state, reward, terminal = env.new_canvas()
        states = []
        actions = []
        v_preds = []
        rewards = []

        for step in range(config['T']):

            act, v_pred = ppo.act(state, stochastic=True)
            act = act[0]
            v_pred = v_pred.item()

            states.append(state)
            actions.append(act)
            v_preds.append(v_pred)
            rewards.append(reward)

            if terminal:
                state, reward, terminal = env.new_canvas()
            else:
                state, reward, terminal = env.draw(act)

        _, v_pred_next = ppo.act(state, stochastic=True)
        v_preds_next = v_preds[1:] + [v_pred_next.item()]

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                           , iteration)

        advantages = ppo.get_advantages(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)
        states = np.array(states)
        actions = np.array(actions)
        v_preds_next = np.array(v_preds_next)
        advantages = np.array(advantages)
        advantages = (advantages-advantages.mean())/advantages.std()

        ppo.assign_parameters()

        inp = [states, actions, rewards, v_preds_next, advantages]

        for epoch in range(config['TRAIN_EPOCH']):
            indices = np.random.randint(low=0, high=states.shape[0], size=config['BATCH_SIZE'])

            input_sample = [np.take(a=a, indices=indices, axis=0) for a in inp]
            ppo.train(state=input_sample[0],
                      actions=input_sample[1],
                      rewards=input_sample[2],
                      v_preds_next=input_sample[3],
                      advantages=input_sample[4])

        summary = ppo.get_summary(state=inp[0],
                                  actions=inp[1],
                                  rewards=inp[2],
                                  v_preds_next=inp[3],
                                  advantages=inp[4])
        writer.add_summary(summary, iteration)

        if iteration % 2000 == 1:
            saver.save(sess, config['SAVED_POLICY'] + '/model.ckpt')
            print('Model saved')

        if iteration % 100 == 0:
            print('Iteration {}, episode reward: {}'.format(iteration, sum(rewards)))

    writer.close()


if __name__ == '__main__':
    sess = tf.Session()
    train(sess)









