"""
Implement operation gate for neural network
operation gate: f = a*x
"""
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# define value
x_val = 5.

# define placeholder
x_data = tf.placeholder(dtype=tf.float32)

# define variable
a = tf.Variable(tf.constant(4.))

# add operation to computational graph
mult = tf.multiply(a, x_data)

# define L2 loss function
loss = tf.square(mult - 50.)

# initialize variable
init = tf.global_variables_initializer()
sess.run(init)

# define optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = m_opt.minimize(loss)

print('Operation Gate: f=a*x')
print('Optimizing a multiplication gate output to 50.')
for i in range(10):
    sess.run(train_step, feed_dict={x_data: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(mult, feed_dict={x_data: x_val})
    print(a_val, ' * ', x_val, ' = ', mult_output)
