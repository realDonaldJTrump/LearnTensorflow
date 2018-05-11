import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
sess = tf.InteractiveSession()

conv_size = 5

def weigth_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding='SAME')

#定义输入数据和标签
x = tf.placeholder(tf.float32, [None, 784])
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#定义网络参数
W_conv1 = weigth_variable([conv_size, conv_size, 1, 32])
b_conv1 = bias_variable([32])
W_conv2 = weigth_variable([conv_size, conv_size, 32, 64])
b_conv2 = bias_variable([64])
W_fc1 = weigth_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weigth_variable([1024, 10])
b_fc2 = bias_variable([10])

#定义网络结构
h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_reshape = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_reshape, W_fc1) + b_fc1)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)
y = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

#定义损失函数
cross_entrophy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
step = tf.train.AdamOptimizer(1e-4).minimize(cross_entrophy)

#评估训练结果
correct_answer = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_answer, tf.float32))

#训练网络
tf.global_variables_initializer().run()
for i in range(10000):
    batch = mnist.train.next_batch(50)
    step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.75})
    entrophy = cross_entrophy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.75})
    if (i+1)%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
        print('第%d轮：交叉熵：%f, 准确率：%f' % ((i+1)/100, entrophy, train_accuracy))

#-----------------------------------------可视化-----------------------------------------------

## 显示图像
fig1,ax1 = plt.subplots(figsize=(2,2))
ax1.imshow(np.reshape(mnist.train.images[11], (28, 28)), cmap=plt.get_cmap('gray'))

## 显示第一层卷积核
W_conv1_reshape = tf.reshape(W_conv1, [32, 1, conv_size, conv_size])
W_conv1_reshape = W_conv1_reshape.eval()
fig2,ax2 = plt.subplots(nrows=4, ncols=8)
for row in range(4):
    for col in range(8): 
        ax2[row][col].imshow(W_conv1_reshape[8*row+col][0], cmap=plt.get_cmap('gray'))

## 显示第二层卷积核
W_conv2_reshape = tf.reshape(W_conv2, [32, 64, conv_size, conv_size])
W_conv2_reshape = W_conv2_reshape.eval()
W_conv2_reshape = W_conv2_reshape[0]
fig,ax = plt.subplots(nrows=8, ncols=8)
for row in range(8):
    for col in range(8): 
        ax[row][col].imshow(W_conv2_reshape[8*row+col], cmap=plt.get_cmap('gray'))

accu = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
print('最终准确率：%f' % accu)
plt.show()

