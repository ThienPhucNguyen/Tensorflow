"""
Working with data sources
"""
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data
from zipfile import ZipFile
import requests
import io
import tarfile
import urllib3
import pandas as pd

# -----------------------------------------------------------
# iris data

print("Iris Data")
iris = datasets.load_iris()

# number of samples
print("number of samples:", len(iris.data))

# number of labels
print("number of labels:", len(iris.target))

# show the label of the first sample
print("first sample:", iris.data[0], "-", iris.target[0], "\n")

# -----------------------------------------------------------
# birth weight data

print("Birth weight data")
birth_data_dir = 'dataset/Low Birthweight Data'

# get dataset from file
f = open(birth_data_dir)
birth_file = pd.read_table(f, sep='\t', index_col=None, lineterminator='\n')
f.close()

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

# -----------------------------------------------------------
# MNIST data

print('MNIST data')

mnist = input_data.read_data_sets('dataset/MNIST_data/', one_hot=True)

# number of train samples
print('number of train samples:', len(mnist.train.images))

# number of test samples
print('number of test samples:', len(mnist.test.images))

# number of validation samples
print('number of validation samples:', len(mnist.validation.images))

# label of the first sample
print('the label of the first label:', mnist.train.labels[1, :], '\n')

# ------------------------------------------------------------
# spam-ham text data

print('spam-ham text data')
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

# get zipfile from url
r = requests.get(zip_url)

# read zip file and store on memory
z = ZipFile(io.BytesIO(r.content))

# get dataset
file = z.read('SMSSpamCollection')
text_data = file.decode()
text_data = text_data.encode('ascii', errors='ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t')for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

# number of train samples
print('number of train samples:', len(text_data_train))

# labels
print('labels:', set(text_data_target))

# the first samples
print('the first samples:', text_data_train[0], '\n')

# -----------------------------------------------------------
# Movie review data

print('movie review data')
movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

r = requests.get(movie_data_url)

# stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
    s = stream_data.read(16384)
    if not s:
        break
    tmp.write(s)
stream_data.close()
tmp.seek(0)

# extract tar file
tar_file = tarfile.open(fileobj=tmp, mode='r:gz')
# positive reviews file
pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
# negative reviews file
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')

# save pos/ng reviews (also deal with encoding)
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())

neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
tar_file.close()

# number of pos data
print('number of pos samples:', len(pos_data))

# number of neg data
print('number of neg samples:', len(neg_data))

# the first negative review
print('the first neg review:', neg_data[0], '\n')

# ----------------------------------------------------------
# the shakespeare text data

print('shakespeare text data')
shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'

# get shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content

# decode binary to string
shakespeare_text = shakespeare_file.decode('utf-8')

# drop the first few paragraphs
shakespeare_text = shakespeare_text[7675:]

# number of characters
print('number of characters:', len(shakespeare_text), '\n')

# ------------------------------------------------------------
# english-german sentence translation data

print('english-german sentence translation data')
sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'

# get dataset
response = requests.get(sentence_url)
# get zip file
z = ZipFile(io.BytesIO(response.content))
# get file
file = z.read('deu.txt')

# format data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii', errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x) >= 1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]

# number of english sentence
print('number of english sentence:', len(english_sentence))

# number of german sentence
print('number of german sentence:', len(german_sentence))

# the first eng-ger sentence pair
print('the first pair:', eng_ger_data[0])