import numpy as np
from skimage.draw import line, bezier_curve
from pretrianGAN.utils import discriminator
import tensorflow as tf
from ppo.config import config
from ppo.utils import move_point


class SketchDiscriminator:
    def __init__(self, path):
        self.X = tf.placeholder(tf.float32, shape=[None] + config['STATE_DIM'] + [1], name='X')
        self.score = discriminator(self.X, rate=1.0)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def inference(self, X):
        X = X.reshape([-1] + config['STATE_DIM'] + [1])
        result = self.sess.run(self.score, feed_dict={self.X: X})
        return X.reshape(config['STATE_DIM']), result

    def get_score(self, X):
        X = X.reshape([-1] + config['STATE_DIM'] + [1])
        _, scores = self.inference(X)
        score = scores[0][0]

        return score


class SketchDesigner:
    def __init__(self, classifier,
                 dim=config['STATE_DIM'], max_stroke=config['MAX_STROKE']):
        self.classifier = classifier
        self.game_count = 0
        self.stroke_count = 0
        self.dim = dim
        self.canvas = np.zeros(self.dim)
        self.max_stroke = max_stroke
        self.terminal = False
        self.previous_score = 0.25

    def draw(self, action):

        # do_nothing, q_line, q_curve, x0_line, y0_line, x1_line ,y1_line,
        # x0_c, y0_c, x1_c, y1_c, x2_c, y2_c, c

        if self.stroke_count >= self.max_stroke - 1:
            self.terminal = True

        # Parameter Validation and noises
        action_category = np.argmax(action[0:3])
        if self.stroke_count == 0:
            axis = np.asarray(action[3:13], dtype=np.uint8) + np.int_(np.random.normal(0, 2, action[3:13].shape[0]))
            c_p = action[13] + np.random.normal(0, 1)
        else:
            axis = np.asarray(action[3:13], dtype=np.uint8)
            c_p = action[13]

        for i in range(axis.shape[0]):
            if axis[i] < 2:
                axis[i] = 2
            elif axis[i] >= self.dim[0] - 2:
                axis[i] = self.dim[0] - 2

        if action_category == 1:
            self.stroke_count += 1
            # Draw line
            rr, cc = line(axis[0], axis[1], axis[2], axis[3])
            self.canvas[rr, cc] = 1

        if action_category == 2:
            self.stroke_count += 1
            # Draw Curve
            try:
                rr, cc = bezier_curve(axis[4], axis[5],
                                  axis[6], axis[7],
                                  axis[8], axis[9],
                                  c_p)
            # TODO: Fix this curve error
            except MemoryError:
                while True:
                    try:
                        _x1, _y1 = move_point(axis[4], axis[5])
                        _x2, _y2 = move_point(axis[6], axis[7])
                        _x3, _y3 = move_point(axis[8], axis[9])
                        rr, cc = bezier_curve(_x1, _y1,
                                              _x2, _y2,
                                              _x3, _y3,
                                              c_p)
                        break
                    except MemoryError:
                        continue

            try:
                self.canvas[rr, cc] = 1
            except IndexError:
                rr = np.clip(rr, 0, config['STATE_DIM'][0] - 1)
                cc = np.clip(cc, 0, config['STATE_DIM'][1] - 1)
                self.canvas[rr, cc] = 1


        score = self.classifier.get_score(self.canvas.reshape(-1, self.dim[0], self.dim[1], 1))

        if score > 0.95:
            self.terminal = True

        if action_category == 0:
            self.terminal = True

        if self.terminal:
            if self.stroke_count == 0:
                reward = -1
            else:
                reward = score - self.previous_score
            self.previous_score = score
        else:
            reward = 0

        return self.canvas, reward, self.terminal

    def new_canvas(self):
        self.canvas = np.zeros(self.dim)
        self.stroke_count = 0
        self.terminal = False

        # print(self.classifier.number_of_goals)

        return self.canvas, 0, self.terminal

    def get_score(self):
        return self.classifier.get_score(self.canvas.reshape(-1, self.dim[0], self.dim[1], 1))




