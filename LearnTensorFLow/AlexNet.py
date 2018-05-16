from datetime import datetime
import math
import time
import tensorflow as tf

# 设置batch_size和训练轮数
batch_size = 32
num_batches = 100

# 显示op的名称和尺寸
def print_tensor(tensor):
    '''显示op的名称和尺寸
    '''
    print(tensor.op.name, tensor.get_shape().as_list())

def inference(images):
    '''正向计算

    @Param:
        images:输入图片batch
    @Return:
        '''
    parameters = [] # 参数列表

    # 第一个卷积层
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[11,11,3,64], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(images, kernel, strides=[1,4,4,1], padding='SAME')
        biases = tf.Variable(tf.constant(value=0.0, shape=[64], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_tensor(conv1)
        parameters += [kernel, biases]
        lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1,alpha=0.001/9,beta=0.75, name='lrn1')
        max_pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='max_pool1')
        print_tensor(max_pool1)

    # 第二个卷积层
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[5,5,64,192], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(max_pool1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(value=0.0, shape=[192], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_tensor(conv2)
        parameters += [kernel, biases]
        lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1,alpha=0.001/9,beta=0.75, name='lrn2')
        max_pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='max_pool2')
        print_tensor(max_pool2)

    # 第三个卷积层
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[3,3,192,384], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(max_pool2, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(value=0.0, shape=[384], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_tensor(conv3)
        parameters += [kernel, biases]

    # 第四个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[3,3,384,384], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(value=0.0, shape=[384], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_tensor(conv4)
        parameters += [kernel, biases]
    
    # 第五个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal(shape=[3,3,384,256], dtype=tf.float32, stddev=0.1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, strides=[1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(value=0.0, shape=[256], dtype=tf.float32), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_tensor(conv5)
        parameters += [kernel, biases]
        max_pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID',name='max_pool5')
        print_tensor(max_pool5)
        return max_pool5, parameters
