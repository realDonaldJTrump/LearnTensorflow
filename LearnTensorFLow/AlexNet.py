from datetime import datetime
import math
import time
import tensorflow as tf

# 设置batch_size和训练轮数
batch_size = 32
num_batches = 100

# 显示op的名称和尺寸
def print_tensor(tensor):
    print(tensor.op.name, tensor.get_shape().as_list())
