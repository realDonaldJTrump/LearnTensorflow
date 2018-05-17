from datetime import datetime
import math
import time
import tensorflow as tf

# 设置batch_size和训练轮数
batch_size = 32
num_batches = 100

def print_tensor(tensor):
    '''显示op的名称和尺寸
    '''
    print(tensor.op.name, tensor.get_shape().as_list())

def inference(images):
    '''正向计算

    @Param:
        images:输入图片batch
    @Returns:
        fc3(tensor):最后一个全连接层的输出
        parameters(list):参数列表'''

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
        #lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1,alpha=0.001/9,beta=0.75, name='lrn1')
        max_pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='max_pool1')
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
        #lrn2 = tf.nn.lrn(conv2, depth_radius=4, bias=1,alpha=0.001/9,beta=0.75, name='lrn2')
        max_pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='max_pool2')
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
    
    # 第一个全连接层
    with tf.name_scope('fc1') as scope:
        max_pool5_reshape = tf.reshape(max_pool5, shape=[batch_size, -1], name='max_pool5_reshape')
        dim = max_pool5_reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 4096], dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[4096]), name='biases')
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(max_pool5_reshape, weights), biases), name=scope)
        print_tensor(fc1)
        parameters += [weights, biases]

    # 第二个全连接层
    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[4096, 4096], dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[4096]), name='biases')
        fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, weights), biases), name=scope)
        print_tensor(fc2)
        parameters += [weights, biases]

    # 第三个全连接层
    with tf.name_scope('fc3') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[4096, 1000], dtype=tf.float32, stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[1000]), name='biases')
        fc3 = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc2, weights), biases), name=scope)
        print_tensor(fc3)
        parameters += [weights, biases]

    return fc3, parameters
    
def time_tensorflow_run(session, target, info_string):
    '''评估每轮计算所需时间

    @Params:
        session: TensorFlow Session
        target: 要进行评估的运算算子
        info_string: 测试的名称
    @Return:
        None
    '''
    num_step_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0 # 总时间的平方和，用于计算方差
    for i in range(num_batches + num_step_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_step_burn_in:
            if i%10 == 0:
                print('%s: step %d, duration: %.3f' % (datetime.now(), i-num_step_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration*duration
    
    mean_duration = total_duration / num_batches
    stdvar_duration = math.sqrt(total_duration_squared / num_batches - mean_duration * mean_duration)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mean_duration, stdvar_duration))

def run_benchmark():
    '''测试AlexNet的计算耗时
    '''
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=0.1))
        fc5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, fc5, 'Forward')

        loss = tf.nn.l2_loss(fc5)
        grad = tf.gradients(loss, parameters)
        time_tensorflow_run(sess, grad, 'Backward')

run_benchmark()
