"""
implement operation gate for neural network
operation gate function: f = a*x + b
"""
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# define value
x_val = 5.

# define placeholder
x_data = tf.placeholder(dtype=tf.float32)

# define Variable
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))

# add operation to the computation graph
mult = tf.multiply(a, x_data)
m_output = tf.add(mult, b)

# define L2 loss function
loss = tf.square(m_output - 50.)

# define optimizer
m_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = m_opt.minimize(loss)

# initialize variable
init = tf.global_variables_initializer()
sess.run(init)

# run loop
print('Optimizing two gate Output to 50.')
for i in range(10):
    # run train step
    sess.run(train_step, feed_dict={x_data: x_val})
    # get the a and b value
    a_val, b_val = (sess.run(a), sess.run(b))
    # run two-gate graph output
    two_gate_output = sess.run(m_output, feed_dict={x_data: x_val})
    print(a_val, ' * ', x_val, ' + ', b_val, ' = ', two_gate_output)