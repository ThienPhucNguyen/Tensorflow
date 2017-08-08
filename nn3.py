"""
Working with gates and activation functions
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start computaiton graph
session = tf.Session()
# set random generator for numpy and tensorflow
tf.set_random_seed(5)
np.random.seed(42)

# define batch size
batch_size = 50

# create data
x_vals = np.random.normal(2, 0.1, 500)

# define placeholder
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# define variables
a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))

# define relu activation function
sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

# define L2 loss function
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

# declare optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step_sigmoid = m_opt.minimize(loss1)
train_trep_relu = m_opt.minimize(loss2)

# initialze variables
init = tf.global_variables_initializer()
session.run(init)

print('\nOptimizing Sigmoid AND Relu Output to 0.75')
loss_vec_sigmoid = []
loss_vec_relu = []
activation_sigmoid = []
activation_relu = []
# run loop
for i in range(500):
    rand_indices = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_indices]])
    session.run(train_step_sigmoid, feed_dict={x_data: rand_x})
    session.run(train_trep_relu, feed_dict={x_data: rand_x})

    loss_vec_sigmoid.append(session.run(loss1, feed_dict={x_data: rand_x}))
    loss_vec_relu.append(session.run(loss2, feed_dict={x_data: rand_x}))

    sigmoid_output = np.mean(session.run(sigmoid_activation, feed_dict={x_data: rand_x}))
    relu_output = np.mean(session.run(relu_activation, feed_dict={x_data: rand_x}))

    if i % 50 == 0:
        print('Step #' + str(i), 'sigmoid = ', sigmoid_output, 'relu = ', relu_output)
    activation_sigmoid.append(sigmoid_output)
    activation_relu.append(relu_output)

# plot the loss and activation output
plt.figure(1)
plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(activation_relu, 'r--', label='ReLU Activation')
plt.ylim([0, 1.])
plt.title('Activation outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')

plt.figure(2)
plt.plot(loss_vec_sigmoid, 'k-', label="Sigmoid Loss")
plt.plot(loss_vec_relu, 'r--', label='ReLU Loss')
plt.ylim([0, 1.])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.show()
