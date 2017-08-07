"""
A simple operations in a computation graph
"""
import tensorflow as tf
import numpy as np

session = tf.Session()

# define placeholders
x_data = tf.placeholder(tf.float32)

# define tensors
x_vals = np.array([1., 3., 5., 7., 9.])
m_const = tf.constant(3.)

# define operation
product = tf.multiply(x_data, m_const)

# show results
for x_val in x_vals:
    print(session.run(product, feed_dict={x_data: x_val}))
