"""
implement back propagation in regression models
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# create graph
sess = tf.Session()

# x-data: 100 random samples from a normal ~ N(1, 0.1)
# target: 100 values of 10
# the model:x-data * A = target
# theoretically, A = 10\

# create data
x_vals = np.random.normal(loc=1, scale=0.1, size=100)
y_vals = np.repeat(a=10., repeats=100)

# create placeholder
x_data = tf.placeholder(dtype=tf.float32, shape=[1])
y_target = tf.placeholder(dtype=tf.float32, shape=[1])

# create variable
A = tf.Variable(tf.random_normal(shape=[1]))

# add multiplication operation to the graph
m_output = tf.multiply(x_data, A)

# add L2 loss function
loss = tf.square(m_output - y_target)

# initialize the variables
init = tf.global_variables_initializer()
sess.run(init)

# define optimize algorithm
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train_step = m_opt.minimize(loss=loss)

# run loop
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))