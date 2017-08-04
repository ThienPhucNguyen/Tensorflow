"""
The very simple program shows how tensorflow work
"""
import tensorflow as tf
import numpy as np

# define the variable
x_val = np.random.rand(2, 2)

# define placeholders used to feed data in the graph
x = tf.placeholder(dtype=tf.float32, shape=[2, 2])
y = tf.identity(input=x)

# create variable initializer
init = tf.global_variables_initializer()

# initialize the graph
with tf.Session() as session:
    # initialize variables in the graph
    session.run(init)
    # run y
    print(session.run(y, feed_dict={x: x_val}))
