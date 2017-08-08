"""
implement one-hidden-layer neural network for regression
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start the graph session
sess = tf.Session()

# load iris data and pedal length as the target value
iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

# set random seed
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# create train/test set = 80/20
train_indices = np.random.choice(len(x_vals), size=round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

# normalize data
x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# initialize placeohlders
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# define hyper parameters
# define batch size
batch_size = 50
# define nodes in each hidden layer
hidden_layer_nodes = 10
# define learning rate
eta = 0.005
# define loop iterations
iter = 500

# create variables for NN layers
# input -> hidden layer
W1 = tf.Variable(tf.random_normal(shape=[3, hidden_layer_nodes]))
# bias1 -> hidden layer
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
# hidden layer -> output
W2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 1]))
# bias2 -> output
b2 = tf.Variable(tf.random_normal(shape=[1]))

# define model operation
# X * W + b
# activation function = ReLU
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, W2), b2))

# define MSE loss function
loss = tf.reduce_mean(tf.square(y_target - final_output))

# define optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=eta)
train_step = m_opt.minimize(loss)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# training loop
loss_vec = []
test_loss = []
for i in range(iter):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x_train = x_vals_train[rand_index]
    rand_y_train = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x_train, y_target: rand_y_train})

    train_temp_loss = sess.run(loss, feed_dict={x_data: rand_x_train, y_target: rand_y_train})
    loss_vec.append(np.sqrt(train_temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(np.sqrt(test_temp_loss))

    if (i + 1) % 50 == 0:
        print('Generation:', (i + 1), 'Loss =', train_temp_loss)

# plot loss MSE over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
