"""
Working with multiple layers to connect various
layers that have data propagating through them
"""
import tensorflow as tf
import numpy as np

session = tf.Session()

# define input data - a 2D image with 4 params
# [image number, height, width, channel]
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

# define placeholder
x_data = tf.placeholder(dtype=tf.float32, shape=x_shape)

# create convolutional layer
# create a sparse moving window average across 4x4 image
# with the shape 2x2
m_filter = tf.constant(value=0.25, shape=[2, 2, 1, 1])
m_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data,
                             m_filter,
                             m_strides,
                             padding='SAME',
                             name='Moving_Avg_Window')

# define a custom layer
def custom_layer(input_matrix):
    input_matrix_squeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_squeezed)
    # Ax + b
    temp2 = tf.add(temp1, b)
    return tf.nn.sigmoid(temp2)

# place the custom layer on the graph
with tf.name_scope('Custom_layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

    # feed in the 4x4 image anf run the graph
    print(session.run(custom_layer1, feed_dict={x_data: x_val}))
