import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time
import math
import matplotlib.pyplot as plt

max_steps = 300
batch_size = 128
data_dir='/tmp/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_loss(shape, stddev, wl=None):
    '''创建变量
    
    创建变量，并为变量添加L2约束'''
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

##下载数据集
cifar10.maybe_download_and_extract()

##生成增强(distorted)过的训练数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
##生成测试数据，只裁剪，无增强
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

##创建输入数据的占位符
images_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, 24, 24, 3])
labels_holder = tf.placeholder(dtype=tf.int32, shape=[batch_size])


#================================↓开始构建正向网络↓==================================

#======================================
#            创建第一个卷积层
#======================================
w_conv1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=0.05)
b_conv1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv_conv1 = tf.nn.conv2d(input=images_holder, filter=w_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1 = tf.nn.relu(tf.nn.bias_add(conv_conv1, b_conv1))
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(h_pool1, depth_radius=4, alpha=0.001/9.0, beta=0.75)

#======================================
#            创建第二个卷积层
#======================================
w_conv2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2)
b_conv2 = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64]))
conv_conv2 = tf.nn.conv2d(norm1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv_conv2, b_conv2))
norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
h_pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#======================================
#            创建第一个全连接层
#======================================
reshape = tf.reshape(h_pool2, shape=[batch_size, -1])
dim = reshape.get_shape()[1].value
w_fc1 = variable_with_weight_loss([dim, 384], stddev=0.04, wl=0.004)
b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1)

#======================================
#            创建第二个全连接层
#======================================
w_fc2 = variable_with_weight_loss([384, 192], stddev=0.04, wl=0.004)
b_fc2 = tf.Variable(tf.constant(0.1, shape=[192]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

#======================================
#            创建输出层
#======================================
w_out = variable_with_weight_loss([192, 10] ,stddev=1/192) #stddev设置为输入节点数的倒数
b_out = tf.Variable(tf.constant(0.0, shape=[10]))
y_out = tf.add(tf.matmul(h_fc2, w_out), b_out)

#================================↑正向网络构建完毕↑==================================

#======================================
#            定义模型损失函数
#======================================
def loss(logits, labels):
    '''计算模型损失

    输入参数：
     logits: 网络输出层的输出
     labels: 训练数据或测试数据的类别标签

    输出：模型的总体损失，包括交叉熵和正则化项'''
    
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

loss = loss(y_out, labels_holder)

##采用Adam优化器，最小化总体损失
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(y_out, labels_holder, 1)

#================================开始训练==================================
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners() #启动线程队列

for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict={images_holder:image_batch, labels_holder:label_batch})
    duration = time.time() - start_time
    if (step+1)%10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_example = float(duration)

        format_str = ('step%d, loss=%.2f (%.1f examples/sec, %.3f sec/batch)')
        real_step = step + 1
        print(format_str % (real_step, loss_value, examples_per_sec, sec_per_example))


#================================评估测试机上的准确率==================================
num_examples = 10000
num_iter = int(math.ceil(num_examples/batch_size))
totle_sample_count = num_iter * batch_size
true_count = 0
for step in range(num_iter):
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], 
                            feed_dict={images_holder:image_batch,
                                       labels_holder:label_batch})
    true_count += np.sum(predictions)

precision = true_count / totle_sample_count
print('precision @ 1 = %.3f' % precision)

