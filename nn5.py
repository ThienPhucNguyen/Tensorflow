"""
implementing with different layers with 1D data
1- Convolution layer
2- Activation layer
3- Max-Pool layer
4- Fully Connected layer
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# ----------------------------------------
# 1D dataset

# start the graph session
sess = tf.Session()

# define parameters
data_size = 25
conv_size = 5
maxpool_size = 5
stride_size = 1

# ensure reproducibility
seed = 13
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# generate 1D data
data_1d = np.random.normal(size=data_size)

# create placeholder
x_data_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

# convolution
def conv_layer_1d(input_1d, filter, stride):
    # tensorflow's 'conv2d()' function only work with 4D arrays:
    # [batch#, width, height, channels],
    # we have 1 batch, width = 1,
    # but height = length of the input and 1 channel
    # So , we create the 4D array by inserting dimension 1's.
    input_2d = tf.expand_dims(input=input_1d, axis=0)
    input_3d = tf.expand_dims(input=input_2d, axis=0)
    input_4d = tf.expand_dims(input=input_3d, axis=3)

    # perform convolution with stride = 1, if we want to increase
    # the stride, say '2' then strides=[1,1,2,1]
    convolution_output = tf.nn.conv2d(input_4d, filter=filter, strides=[1, 1, stride, 1], padding='VALID')

    # get rid of extra dimension
    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d

# create filter for convolution
m_filter = tf.Variable(tf.random_normal(shape=[1, conv_size, 1, 1]))

# create convolution layer
m_convolution_output = conv_layer_1d(x_data_1d, m_filter, stride_size)

# activation
def activation(input_1d):
    return tf.nn.relu(input_1d)

# create activation layer
m_activation_output = activation(m_convolution_output)

# max pool
def max_pool(input_1d, width, stride):
    # just like 'conv2()', 'max_pool()' works with 4d arrays.
    # [batch=1, width=1, height=num_input, channels=1]
    input_2d = tf.expand_dims(input_1d, axis=0)
    input_3d = tf.expand_dims(input_2d, axis=0)
    input_4d = tf.expand_dims(input_3d, axis=3)

    # perform max pooling with stride = [1,1,1,1]
    # if we want to increase the stride on our data dimension,
    # put strides=[1,1,2,1]
    # we also need to specify the width of the max dimension ('width')
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, 1, width, 1],
                                 strides=[1, 1, stride, 1],
                                 padding='VALID')
    # get rid of extra dimensions
    pool_output_1d = tf.squeeze(pool_output)
    return pool_output_1d

# create max pool layer
m_maxpool_output = max_pool(m_activation_output, maxpool_size, stride_size)

# full connected
def fully_connected(input_layer, num_outputs):
    # find the needed shape of the multiplication weight matrix:
    # the dimension will be (length of input) by (num_outputs)
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer), [num_outputs]]))

    # initialize such weight
    weight = tf.random_normal(shape=weight_shape, stddev=0.1)

    # initialize bias
    bias = tf.random_normal(shape=[num_outputs])

    # make the 1D array into 2D array for matrix multiplication
    input_layer_2d = tf.expand_dims(input_layer, axis=0)

    # perform a matrix multiplication and add the bias
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)

    # get rid of extra dimension
    full_output_1d = tf.squeeze(full_output)
    return full_output_1d

m_full_output = fully_connected(m_maxpool_output, 5)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

feed_dict = {x_data_1d: data_1d}

print('>>>>>1D Data<<<<<')

# convolution output
print('Input = array of length %d' % (x_data_1d.shape.as_list()[0]))
print('Convolution w/ filter, length = %d, stride size = %d, results in an array of length %d:' %
      (conv_size, stride_size, m_convolution_output.shape.as_list()[0]))
print(sess.run(m_convolution_output, feed_dict=feed_dict))

# activation output
print('\nInput = above array of length %d' % m_convolution_output.shape.as_list()[0])
print('ReLU element wise returns an array of length %d:' % m_activation_output.shape.as_list()[0])
print(sess.run(m_activation_output, feed_dict=feed_dict))

# max pooling output
print('\nInput = above array of length %d' % m_maxpool_output.shape.as_list()[0])
print('Fully connected layer on all 4 rows with %d outputs:' % m_full_output.shape.as_list()[0])
print(sess.run(m_full_output, feed_dict=feed_dict))