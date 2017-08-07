"""
implement the back propagaton for classification
models
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()

# create the graph
sess = tf.Session()

# create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

# create placeholders
x_data = tf.placeholder(dtype=tf.float32, shape=[1])
y_target = tf.placeholder(dtype=tf.float32, shape=[1])

# create variable
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# add operation to graph
# create a operation sigmoid(x + A)
# sigmoid is a part in loss function
m_output = tf.add(x_data, A)

# add another dimension to each (batch size of 1)
m_output_expanded = tf.expand_dims(m_output, 0)
y_target_expanded = tf.expand_dims(y_target, 0)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target_expanded,
                                                   logits=m_output_expanded)

# create optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_step = m_opt.minimize(xentropy)

# run loop
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1) % 200 == 0:
        print('Step#' + str(i + 1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))

# evaluate predictions
predictions = []

for i in range(len(x_vals)):
    x_val = [x_vals[i]]
    prediction = sess.run(tf.round(tf.sigmoid(m_output)), feed_dict={x_data: x_val})
    predictions.append(prediction[0])

accuracy = sum(x == y for x,y in zip(predictions, y_vals)) / 100
print('Ending Accuracy = ', np.round(accuracy, 2))