"""
Putting multiple operations on the same computation
"""
import tensorflow as tf
import numpy as np

# create input data
m_array = np.array([[1., 3., 5., 7., 9.],
                    [-2, 0., 2., 4., 6.],
                    [-6, -3., 0., 3., 6.]])

x_vals = np.array([m_array, m_array + 1])

# define placeholders
x_data = tf.placeholder(dtype=tf.float32, shape=(3, 5))

# create constants
m1 = tf.constant([[1.], [0.], [-1.], [2.], [5.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# define operations and ass to the graph
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # print result
    for x_val in x_vals:
        print(sess.run(add1, feed_dict={x_data: x_val}))


# feed data through the graph

