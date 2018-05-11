import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class DenoisingAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), noise_scale=0.1):
        
        ##初始化参数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(dtype=tf.float32)
        self.training_scale = noise_scale
        self.weights = self._initialize_weights()

        ##定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        noised_input = self.x + noise_scale * tf.random_normal((n_input,)) #(n_input,)是只有一个值的元组
        self.hidden = self.transfer(tf.add(tf.matmul(noised_input, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        self.sess = tf.Session()
        
    def _initialize_weights(self):
        weights = {}
        return weights


if __name__ == '__main__':
    test = tf.random_normal([10])
    sess = tf.Session()
    print(sess.run(test))
    