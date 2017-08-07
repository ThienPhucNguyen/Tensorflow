"""
working batch and stochastic training in a regression model
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start the computation graph
sess = tf.Session()

# -------------------------------------------------------------
# stochastic gradient descent

# create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

# create placeholder
x_data = tf.placeholder(dtype=tf.float32, shape=[1])
y_target = tf.placeholder(dtype=tf.float32, shape=[1])

# create variable
A = tf.Variable(tf.random_normal(shape=[1]))

# add operation to the graph
m_output = tf.multiply(x_data, A)

# add L2 loss operation to the graph
loss = tf.square(m_output - y_target)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# define GD optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = m_opt.minimize(loss)

loss_stochastic = []

# run loop
for i in range(200):
    if i % 100 == 0:
        x_vals = np.shuffle
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)

# -------------------------------------------------------------
# batch gradient descent (mini-batch)
ops.reset_default_graph()

sess = tf.Session()

# define batch size
batch_size = 20

# create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

# create placeholders
x_data = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y_target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# create variable
A = tf.Variable(tf.random_normal(shape=[1, 1]))

# add operation to the graph
m_output = tf.matmul(x_data, A)

# add L2 loss operation to the graph
# mean loss value
loss = tf.reduce_mean(tf.square(m_output - y_target))

# initialize variable
init = tf.global_variables_initializer()
sess.run(init)

# create the GD optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = m_opt.minimize(loss)

loss_batch = []

# run loop
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 5 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

# visualize loss
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size = 20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()