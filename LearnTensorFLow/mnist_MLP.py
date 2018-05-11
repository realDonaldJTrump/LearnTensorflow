import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)


##定义输入层和隐藏层维度
in_units = 784
h1_units = 200
h2_units = 100
    
##创建参数和偏置变量
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
b2 = tf.Variable(tf.zeros([h2_units]))
W3 = tf.Variable(tf.zeros([h2_units, 10]))
b3 = tf.Variable(tf.zeros([10]))

##定义inference过程
x = tf.placeholder(tf.float32, [None ,in_units], 'input')
keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
hidden_drop = tf.nn.dropout(hidden2, keep_prob=keep_prob)
y = tf.nn.softmax(tf.matmul(hidden_drop, W3) + b3)
y_ = tf.placeholder(tf.float32, [None, 10])

    ##定义代价函数
with tf.Session() as sess:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.AdagradOptimizer(0.2).minimize(cross_entropy)
    #train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    ##开始训练
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        bathc_xs, batch_ys = mnist.train.next_batch(100)
        _, entrophy = sess.run([train_step, cross_entropy], feed_dict={x:bathc_xs, y_:batch_ys, keep_prob:0.6})
        if i%500 == 0:
            print('第%d次交叉熵: %f' % (i, entrophy))
    
    ##评估训练结果
    correct_result = tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_result)
    accu = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
    print('准确度: ', accu)
