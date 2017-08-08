"""
implement with different with 2D data
1- Convolution layer
2- Activation layer
3- Max-Pool layer
4- Fully Connected layer
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start graph session
sess = tf.Session()

# parameters for the run
row_size = 10
col_size = 10
conv_size = 2
conv_stride_size = 2
maxpool_size = 2
maxpool_stride_size = 1

# ensure reproducibility
seed = 13
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# generate 2D data
data_size = [row_size, col_size]
data_2d = np.random.normal(size=data_size)

# create placeholder
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

# convolution function
def conv_layer_2d(input_2d, filter, stride):
    # tensorflow's 'conv2d()' function ONLY work with 4d arrays.
    # [batch#, width, height, channels]
    input_3d = tf.expand_dims(input_2d, axis=0)
    input_4d = tf.expand_dims(input_3d, axis=-1)

    convolution_output = tf.nn.conv2d(input_4d,
                                      filter=filter,
                                      strides=[1, stride, stride, 1],
                                      padding='VALID')

    # get rid of extra dimension
    conv_output_2d = tf.squeeze(convolution_output)
    return conv_output_2d

# create convolution filter
m_filter = tf.Variable(tf.random_normal(shape=[conv_size, conv_size, 1, 1]))

# create convolution layer
m_convolution_output = conv_layer_2d(x_input_2d, m_filter, conv_stride_size)

# activation function
def activation(input_1d):
    return tf.nn.relu(input_1d)

# create activation layer
m_activation_output = activation(m_convolution_output)

# mac pooling function
def maxpool(input_2d, width, height, stride):
    # just like 'conv2d()', 'max_pool()' ONLY works with 4d arrays.
    # [batch-size=1, width=given, height=given, channels=1]
    input_3d = tf.expand_dims(input_2d, axis=0)
    input_4d = tf.expand_dims(input_3d, axis=-1)

    # perform the max pooling with stride=[1,1,1,1]
    maxpool_output = tf.nn.max_pool(input_4d,
                                    ksize=[1, height, width, 1],
                                    strides=[1, stride, stride, 1],
                                    padding='VALID')

    # get rid of extra dimension
    pool_output_2d = tf.squeeze(maxpool_output)
    return pool_output_2d

# create max pooling layer
m_maxpool_output = maxpool(m_activation_output, maxpool_size, maxpool_size, maxpool_stride_size)

# fully connected function
def fully_connected(input_layer, num_output):
    # in order to connect W x H 2d array --> flatten it out to W x H 1d array
    flat_input = tf.reshape(input_layer, [-1])

    # define weight shape
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_output]]))

    # initialize weights
    weight = tf.random_normal(shape=weight_shape, stddev=0.1)

    # initialize the bias
    bias = tf.random_normal(shape=[num_output])

    # now make the flat 1d array into a 2d array for multiplication
    input_2d = tf.expand_dims(flat_input, 0)

    # multiply and add the bias
    full_output = tf.add(tf.matmul(input_2d, weight), bias)

    # get rid of extra dimension
    full_output_2d = tf.squeeze(full_output)
    return full_output_2d

# create fully connected layer
m_full_output = fully_connected(m_maxpool_output, 5)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_input_2d: data_2d}

print('>>>>>2D data<<<<<')
# Convolution Output
print('Input = %s array' % (x_input_2d.shape.as_list()))
print('%s Convolution, stride size = [%d, %d] , results in the %s array' %
      (m_filter.get_shape().as_list()[:2],conv_stride_size, conv_stride_size, m_convolution_output.shape.as_list()))
print(sess.run(m_convolution_output, feed_dict=feed_dict))

# Activation Output
print('\nInput = the above %s array' % (m_convolution_output.shape.as_list()))
print('ReLU element wise returns the %s array' % (m_activation_output.shape.as_list()))
print(sess.run(m_activation_output, feed_dict=feed_dict))

# Max Pool Output
print('\nInput = the above %s array' % (m_activation_output.shape.as_list()))
print('MaxPool, stride size = [%d, %d], results in %s array' %
      (maxpool_stride_size,maxpool_stride_size, m_maxpool_output.shape.as_list()))
print(sess.run(m_maxpool_output, feed_dict=feed_dict))

# Fully Connected Output
print('\nInput = the above %s array' % (m_maxpool_output.shape.as_list()))
print('Fully connected layer on all %d rows results in %s outputs:' %
      (m_maxpool_output.shape.as_list()[0], m_full_output.shape.as_list()[0]))
print(sess.run(m_full_output, feed_dict=feed_dict))


