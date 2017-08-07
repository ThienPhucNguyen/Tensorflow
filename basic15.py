"""
working with evaluating simple regression models (mini batch DG)
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start the graph session
sess = tf.Session()

# define batch size
batch_size = 25

# create data
x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

# create placeholders
x_data = tf.placeholder(dtype=tf.float32, shape=[1, None])
y_target = tf.placeholder(dtype=tf.float32, shape=[1, None])

# split data into train/test = 80/20
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8))
test_indices = (list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# create variable
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# add operation to graph
# create sigmoid(x + A)
m_output = tf.add(x_data, A)

# create sigmoid loss function (cross entropy)
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target,
                                                                  logits=m_output))

# create optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = m_opt.minimize(xentropy)

# initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# run loop
for i in range(1800):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = [x_vals_train[rand_index]]
    rand_y = [y_vals_train[rand_index]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1 ) % 200 == 0:
        print('Step #' + str(i + 1) + ' a = ' + str(sess.run(A)))
        print('Loss =', sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y}))

# evaluate prediction on test set
y_prediction = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(x_data, A))))
correct_prediction = tf.equal(y_prediction, y_target)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_value_test = sess.run(accuracy, feed_dict={x_data: [x_vals_test], y_target: [y_vals_test]})
acc_value_train = sess.run(accuracy, feed_dict={x_data: [x_vals_train], y_target: [y_vals_train]})

print('Accuracy on train set:', acc_value_train)
print('Accuracy on test set:', acc_value_test)

# plot the classification result
A_result = -sess.run(A)
bins = np.linspace(-5, 5, 50)
plt.hist(x=x_vals[0: 50],
         bins=bins,
         alpha=0.5,
         label='N(-1,1)',
         color='yellow')
plt.hist(x=x_vals[50: 100],
         bins=bins[0: 50],
         alpha=0.5,
         label='N(2,1)',
         color='red')
plt.plot((A_result, A_result), (0, 8), 'k--', linewidth=3, label='A = ' + str(np.round(A_result, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy=' + str(np.round(acc_value_test, 2)))
plt.show()
