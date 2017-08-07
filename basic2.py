"""
Working with matrix
"""
import tensorflow as tf
import numpy as np

# initialize the graph
from tensorflow.python.ops.gen_dataset_ops import tensor_dataset

session = tf.Session()

#----------------------------------------
# create matrices

# create an identity matrix
identity_matrix = tf.diag(diagonal=[1., 1., 1.])

# create matrix from random tensor with truncated
# normal distribution
A = tf.truncated_normal(shape=[2, 3])

# create matrix from fixed tensor with certain values
B = tf.fill(dims=[2, 3], value=5.)

# create matrix from uniform distribution
C = tf.random_uniform(shape=[3, 2])

# create matrix from numpy array
D = tf.convert_to_tensor(np.array([[1., 2., 3.],[-3., -7., -1.],[0., 5., -2.]]))

# show the matrices
print("identity matrix \n", session.run(identity_matrix), "\n")
print("random truncated normal matrix \n", session.run(A), "\n")
print("certain value matrix \n", session.run(B), "\n")
print("random uniform matrix \n", session.run(C), "\n")
print("numpy array based matrix \n", session.run(D), "\n")

# -------------------------------------------
# matrix operations

# addition and subtraction
print("addition operation\n", session.run(A + B), "\n")
print("substraction operation\n", session.run(A - B), "\n")
# multiplication
print("multiply operation\n", session.run(tf.matmul(B, identity_matrix)), "\n")

# transpose the matrix
print("transpose operation\n", session.run(tf.transpose(a=C)), "\n")

# matrix determinant
print("matrix determinant\n", session.run(tf.matrix_determinant(input=D)), "\n")

# inverse matrix
print("inverse operation\n", session.run(tf.matrix_inverse(input=D)), "\n")

# matrix decomposition

# cholesky decomposition
print("cholesky decomposition\n", session.run(tf.cholesky(input=identity_matrix)), "\n")

# eigenvalues and eigenvectors of the matrix
print("eigen decomposition", session.run(tf.self_adjoint_eig(tensor=D)), "\n")
print("eigen values", session.run(tf.self_adjoint_eigvals(tensor=D)), "\n")

