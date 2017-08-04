"""
Working with data sources
"""
from sklearn import datasets
import pandas as pd

#-----------------------------------------------------------
# iris data

print("Iris Data")
iris = datasets.load_iris()

# number of samples
print("number of samples:", len(iris.data))

# number of labels
print("number of labels:", len(iris.target))

# show the label of the first sample
print("first sample:", iris.data[0], "-", iris.target[0], "\n")

#-----------------------------------------------------------
# birth weight data

print("Birth weight data")
birth_data_dir = 'dataset/Low Birthweight Data'

# get dataset from file
f = open(birth_data_dir)
birth_file = pd.read_table(f, sep='\t', index_col=None, lineterminator='\n')

print(birth_file)
print(birth_file.values)

# get dataset
birth_data = birth_file.values

# get header
birth_header = birth_file.columns.values


# number of samples
print("number of samples:", len(birth_data))

# number of labels
print("number of labels:", len(birth_header), "\n")



