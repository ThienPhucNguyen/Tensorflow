"""
Making a simple binary classifier
on iris dataset
Using mini-batch gradient descent
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn import datasets
ops.reset_default_graph()

# load iris data
# iris.target = {0, 1, 2} where '0' is setosa
# iris.data ~ [sepal.width, sepal.length, pedal.width, pedal.length]
iris = datasets.load_iris()
binary_target = np.array([1. if x == 0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

# define batch size
batch_size = 20

# start the graph
sess = tf.Session()

# create placeholders
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# create variable A and b (0 = x1 - A*x2 + b)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# add model to the graph
m_mul = tf.matmul(x2_data, A)
m_add = tf.add(m_mul, b)
m_output = tf.subtract(x1_data, m_add)

# add cross entropy loss
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=m_output, labels=y_target)

# create optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = m_opt.minimize(xentropy)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# run loop
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))

# visualize results
# pull out slope/intercept
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

# create fitted line
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append(slope * i + intercept)

# plot the fitted line over the data
setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0]

plt.plot(setosa_x, setosa_y, 'r.', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'r+', label='non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0., 2.7])
plt.ylim([0., 7.1])
plt.suptitle('Linear Separator for I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()
