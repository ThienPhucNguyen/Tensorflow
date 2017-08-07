"""
working with evaluating simple regression models (mini batch DG)
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start the graph session
sess = tf.Session()

# define batch size
batch_size = 25

# create data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)

# create placeholders
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# split data into train/test = 80/20
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# create model variable A
A = tf.Variable(tf.random_normal(shape=[1, 1]))

# add operation to graph
m_output = tf.matmul(x_data, A)

# add L2 loss
loss = tf.reduce_mean(tf.square(m_output - y_target))

# create optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = m_opt.minimize(loss)

# initialize variable
init = tf.global_variables_initializer()
sess.run(init)

# run loop
for i in range(100):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 25 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))

# evaluate regression model
# evaluate on test set
mse_test = sess.run(loss,
                    feed_dict={x_data: np.transpose([x_vals_test]),
                               y_target: np.transpose([y_vals_test])})

mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]),
                                      y_target: np.transpose([y_vals_train])})

print('MSE on test:', np.round(mse_test, 2))
print('MSE on train:', np.round(mse_train, 2))