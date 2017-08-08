"""
implement a multilayer neural network
with Low birthweight dataset
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
ops.reset_default_graph()

# make results reproducible
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

# start the graph session
sess = tf.Session()

# load data
# Columns Variable                                      Abbreviation
# ---------------------------------------------------------------------
# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW (LABEL)
#                          1 = Birth Weight < 2500g)
# Age of the Mother in Years                              AGE
# Weight in Pounds at the Last Menstrual Period           LWT
# Race (1 = White, 2 = Black, 3 = Other)                  RACE
# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
# History of Premature Labor (0 = None  1 = One, etc.)    PTL
# History of Hypertension (1 = Yes, 0 = No)               HT
# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
# Birth Weight in Grams                                   BWT (REGRESSION OUTPUT)
# ---------------------------------------------------------------------
birth_data_dir = 'dataset/Low Birthweight Data'
birth_file = open(birth_data_dir)
birth_table = pd.read_table(birth_file, sep='\t',
                            index_col=None,
                            lineterminator='\n')
birth_file.close()

birth_data = birth_table.values
birth_header = birth_table.columns.values

# create data
x_vals = np.array([[float(x[idx]) for idx in range(1, len(birth_header) - 1)] for x in birth_data])
y_vals = np.array([float(x[8]) for x in birth_data])

# split data into train/test = 80/20
train_indices = np.random.choice(len(x_vals), size=round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

# normalize data by column (min-max norm to be between 0 and 1)
def normalize_cols(mat):
    col_max = mat.max(axis=0)
    col_min = mat.min(axis=0)
    return (mat - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# define hyper parameters
batch_size = 100
eta = 0.05
iteration = 200
hidden_node_1 = 25
hidden_node_2 = 20
hidden_node_3 = 3

# define variable function (weights and bias)
def init_weight(shape, stddev):
    weight = tf.Variable(tf.random_normal(shape=shape, stddev=stddev))
    return weight

def init_bias(shape, stddev):
    bias = tf.Variable(tf.random_normal(shape=shape, stddev=stddev))
    return bias

# create placeholder
x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# create a fully connected layer
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.relu(layer)

# create the first layer
W1 = init_weight(shape=[7, hidden_node_1], stddev=10.)
b1 = init_bias(shape=[hidden_node_1], stddev=10.)
layer1 = fully_connected(x_data, W1, b1)

# create the second layer (25 hidden nodes)
W2 = init_weight(shape=[hidden_node_1, hidden_node_2], stddev=10.)
b2 = init_bias(shape=[hidden_node_2], stddev=10.)
layer2 = fully_connected(layer1, W2, b2)

# create the third layer (5 hidden nodes)
W3 = init_weight(shape=[hidden_node_2, hidden_node_3], stddev=10.)
b3 = init_bias(shape=[hidden_node_3], stddev=10.)
layer3 = fully_connected(layer2, W3, b3)

# create output layer (1 output value)
W4 = init_weight(shape=[hidden_node_3, 1], stddev=10.)
b4 = init_bias(shape=[1], stddev=10.)
final_output = fully_connected(layer3, W4, b4)


# define loss function (L1)
loss = tf.reduce_mean(tf.abs(final_output - y_target))

# define optimizer
opt = tf.train.AdamOptimizer(learning_rate=eta)
train_step = opt.minimize(loss)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# run loop
loss_vec = []
test_loss = []
for i in range(iteration):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(test_temp_loss)
    if (i + 1) % 25 == 0:
        print('Generation:', (i + 1), 'Loss =', temp_loss)

# plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
