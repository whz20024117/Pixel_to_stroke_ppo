import tensorflow as tf
from ppo.config import config


class Policy:
    def __init__(self, name:str):

        with tf.variable_scope(name):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None] + config['STATE_DIM'] + [1])

            with tf.variable_scope('shared_cnn', reuse=False):
                x = tf.layers.conv2d(self.state, filters=32, kernel_size=[7, 7], strides=(2, 2), padding='same',
                                     name='conv1',activation=tf.nn.relu)
                x = tf.layers.conv2d(x, filters=64, kernel_size=[2, 2], strides=(1, 1), padding='same',
                                     name='conv2', activation=tf.nn.relu)
                x = tf.layers.max_pooling2d(x, [2, 2],strides=[2, 2], padding='same',
                                            name='pool1')
                x = tf.layers.conv2d(x, filters=128, kernel_size=[2, 2], strides=(1, 1), padding='same',
                                     name='conv3',activation=tf.nn.relu)
                cnn_policy = tf.layers.flatten(tf.layers.max_pooling2d(x, [2, 2], strides=[2, 2], padding='same',
                                            name='pool2'))

            with tf.variable_scope('policy_net'):
                x = tf.layers.dense(cnn_policy, 256, activation=tf.nn.relu, name='fc1')
                x = tf.layers.dense(x, 128, activation=tf.nn.relu, name='fc2')
                self.action_pred = tf.layers.dense(x, config['ACTION_DIM'], name='output', activation=tf.nn.tanh)

            with tf.variable_scope('shared_cnn', reuse=True):
                x = tf.layers.conv2d(self.state, filters=32, kernel_size=[7, 7], strides=(2, 2), padding='same',
                                     name='conv1',activation=tf.nn.relu)
                x = tf.layers.conv2d(x, filters=64, kernel_size=[2, 2], strides=(1, 1), padding='same',
                                     name='conv2', activation=tf.nn.relu)
                x = tf.layers.max_pooling2d(x, [2, 2],strides=[2, 2], padding='same',
                                            name='pool1')
                x = tf.layers.conv2d(x, filters=128, kernel_size=[2, 2], strides=(1, 1), padding='same',
                                     name='conv3',activation=tf.nn.relu)
                cnn_value = tf.layers.flatten(tf.layers.max_pooling2d(x, [2, 2], strides=[2, 2], padding='same',
                                            name='pool2'))

            with tf.variable_scope('value_net'):
                x = tf.layers.dense(cnn_value, 256, activation=tf.nn.relu, name='fc1')
                x = tf.layers.dense(x, 128, activation=tf.nn.relu, name='fc2')
                self.value_pred = tf.layers.dense(x, 1, name='output')

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
