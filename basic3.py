"""
Some examples about standard and
non - standard operations for tensors
"""
import tensorflow as tf

session = tf.Session()

# the floor of division if inputs are integers
print("3 / 4 =", session.run(tf.div(3, 4)))

# the true result of division
print("3 / 4 =", session.run(tf.truediv(3, 4)))

# the floor of division
print("3.0 / 4.0 =", session.run(tf.floordiv(3., 4.)))

# mod operation
print("22.0 mod 5.0 =", session.run(tf.mod(22.0, 5.0)))

# cross product operaiton
print("cross product of [1., 0., 0.] and [0., 1., 0.]\n",
      session.run(tf.cross([1., 0., 0.], [0., 1., 0.])), "\n")

# self-defined tangent function (tan(pi / 4) = 1)
print("tan(pi / 4) =", session.run(tf.div(tf.sin(3.1416 / 4.), tf.cos(3.1416 / 4.))))
