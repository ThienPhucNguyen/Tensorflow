"""
Some built-in activation functions
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

ops.reset_default_graph()

session = tf.Session()

# print value
print("value:", [-3., 3., 10.])

# ReLU = max(0, x)
print("ReLU function:", session.run(tf.nn.relu([-3., 3., 10.])))

# ReLU6 = min(max(0, x),6)
print("ReLU6 function:", session.run(tf.nn.relu6([-3., 3., 10.])))

# print value
print("\nvalue:", [-1., 0., 1.])

# sigmoid = 1 / (1 + exp(-x))
print("sigmoid function:", session.run(tf.nn.sigmoid([-1., 0., 1.])))

# softsign = x / (abs(x) + 1)
print("softsign function:", session.run(tf.nn.softsign([-1., 0., 1.])))

# softplus = log(exp(x) + 1)
print("softplus function:", session.run(tf.nn.softplus([-1., 0., 1.])))

# ELU = (exp(x) + 1)if x < 0 else x
# ReLU = max(0, x)
print("ELU function:", session.run(tf.nn.elu([-1., 0., 1.])))

#-----------------------------------------------------------
# plot the functions

# initialize x range value for plotting
x_vals = np.linspace(start=-10., stop=10., num=100)

y_relu = session.run(tf.nn.relu(x_vals))
y_relu6 = session.run(tf.nn.relu6(x_vals))
y_sigmoid = session.run(tf.nn.sigmoid(x_vals))
y_tanh = session.run(tf.nn.tanh(x_vals))
y_softsign = session.run(tf.nn.softsign(x_vals))
y_softplus = session.run(tf.nn.softplus(x_vals))
y_elu = session.run(tf.nn.elu(x_vals))

# plotting
plt.figure(1)
plt.plot(x_vals, y_softplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x_vals, y_relu, 'b:', label='ReLU', linewidth=2)
plt.plot(x_vals, y_relu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x_vals, y_elu, 'k-', label='ELU', linewidth=1)
plt.ylim([-1.5, 7])
plt.legend(loc='upper left')

plt.figure(2)
plt.plot(x_vals, y_sigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x_vals, y_tanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x_vals, y_softsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2, 2])
plt.legend(loc='upper left')

plt.show()

# keep all figures alive --> use plt.figure(idx)