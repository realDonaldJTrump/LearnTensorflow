import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# from keras.datasets import mnist
# import matplotlib.pyplot as plt

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# plt.imshow(X_train[111], cmap=plt.get_cmap('PuBuGn_r'))
# plt.show()

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entrofy = -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1])

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entrofy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


print("Train Accuracy: "+repr(sess.run(accuracy, feed_dict={x:mnist.train.images, y_:mnist.train.labels})))
print("Test Accuracy: "+repr(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})))

sess.close
