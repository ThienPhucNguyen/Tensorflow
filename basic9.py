"""
Implement loss functions (used to measure the distance between
the model outputs and the target values)
"""
import tensorflow as tf
import matplotlib.pyplot as plt

session = tf.Session()
# -------------------------------------------------------------
# numerical predictions
x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# L2 norm loss = 1 / 2 * (target - pred) ^ 2
l2_y_vals = tf.square(target - x_vals)
l2_y_out = session.run(l2_y_vals)

# L1 norm loss = abs(target - pred)
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = session.run(l1_y_vals)

# Pseudo-Huber loss = delta^2 * (sprt(1 + ((pred - target) / delta)^2) - 1)
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1),
                             tf.sqrt(1. + tf.square(target - x_vals) / delta1) - 1.)
phuber1_y_out = session.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2),
                             tf.sqrt(1. + tf.square(target - x_vals) / delta2) - 1.)
phuber2_y_out = session.run(phuber2_y_vals)

# plot the output
x_array = session.run(x_vals)
plt.figure(1)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label='PHuber Loss (0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label='PHuber Loss (5.0)')
plt.ylim(-2.0, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
#plt.show()

# -------------------------------------------------------------
# categorical predictions
x_vals = tf.linspace(-3., 5, 500)
target = tf.constant(1.)
targets = tf.fill([500, ], 1.)

# Hinge loss = max(0, 1 - (target * pred))
hinge_y_vals = tf.maximum(0.,
                          1. - tf.multiply(target, x_vals))
hinge_y_out = session.run(hinge_y_vals)

# cross entropy loss = -target * log(pred) - (1 - target) * log(1 - pred)
xentropy_y_vals = tf.multiply(-target, tf.log(x_vals)) -\
    tf.multiply(1. - target, tf.log(1. - x_vals))
xentropy_y_output = session.run(xentropy_y_vals)

# sigmoid cross entropy loss
x_val_input = tf.expand_dims(x_vals, 1)
target_input = tf.expand_dims(targets, 1)
xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_val_input, labels=target_input)
xentropy_sigmoid_y_output = session.run(xentropy_sigmoid_y_vals)

# weighted cross entropy loss = -target * log(pred) * weight - (1 - target) * log(1 - pred)
# or
# weighted cross entropy loss = (1 - pred) * target + (1 + (weight - 1) * pred) * log(1 + exp(-target))
weight = tf.constant(0.5)
xentropy_weighted_y_vals = tf.nn.weighted_cross_entropy_with_logits(targets=targets, logits=x_vals, pos_weight=weight)
xentropy_weighted_y_output = session.run(xentropy_weighted_y_vals)

# plot the output
x_array = session.run(x_vals)
plt.figure(2)
plt.plot(x_array, hinge_y_out, 'b-', label='Hinge Loss')
plt.plot(x_array, xentropy_y_output, 'r--', label='Cross Entropy Loss')
plt.plot(x_array, xentropy_sigmoid_y_output, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(x_array, xentropy_weighted_y_output, 'g:', label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()

# ------------------------------------------------------------
# soft-max entropy and sparse entropy (multiclass loss functions)

# soft-max entropy loss = -target * log(softmax(pred)) - (1 - target) * log(1 - softmax(pred))
unscaled_logits = tf.constant([[1., -3., 10.]])
target_dist = tf.constant([[0.1, 0.02, 0.88]])
softmax_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=unscaled_logits,
                                                          labels=target_dist)
print(session.run(softmax_entropy))

# sparse entropy loss
# when classes and target have to be mutually exclusive (loai tru lan nhau)
unscaled_logits = tf.constant([[1., -3., 10.]])
sparse_target_dist = tf.constant([2])
sparse_xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sparse_target_dist,
                                                                 logits=unscaled_logits)
print(session.run(sparse_xentropy))