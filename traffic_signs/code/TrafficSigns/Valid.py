import pickle
from sklearn.model_selection import train_test_split
import random
import csv
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from Helper import LeNet

training_file = '/home/he/prj/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
testing_file = '/home/he/prj/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

with open('/home/he/prj/CarND-Traffic-Sign-Classifier-Project/signnames.csv', 'r') as csvfile:
    signreader = csv.reader(csvfile, delimiter=',')
    signnames = list(signreader)

assert (len(X_train) == len(y_train))
assert (len(X_valid) == len(y_valid))
assert (len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

X_train, y_train = shuffle(X_train, y_train)

count = tf.Variable(0, dtype=tf.int32)
accuracy_value = tf.Variable(0.0, dtype=tf.float64)
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 42)

logits = LeNet(x)
sol1 = tf.nn.relu(logits)
sol2 = tf.argmax(sol1, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'data/lenet.ckpt')
    print("count:" + str(sess.run(count)) + " validation_accuracy:" + str(sess.run(accuracy_value)))

    num_examples = len(X_train)

    image = []
    image.append(X_train[5])
    image.append(X_train[10])
    image.append(X_train[15])
    image.append(X_train[20])
    res = sess.run(sol2, feed_dict={x: image})
    print("prediction:" + str(res))
    print("prediction:" + str(signnames[res[0] + 1]) + " " + str(signnames[res[1] + 1]) + " " +
          str(signnames[res[2] + 1]) + " " + str(signnames[res[3] + 1]))
    print("actual:" + str(signnames[y_train[5] + 1]) + " " + str(signnames[y_train[10] + 1]) + " " +
          str(signnames[y_train[15] + 1]) + " " + str(signnames[y_train[20] + 1]))

    plt.figure(figsize=(1, 1))
    plt.imshow(image[0].squeeze(), cmap="gray")
    plt.show()  # waits for window to be closed
    plt.figure(figsize=(1, 1))
    plt.imshow(image[1].squeeze(), cmap="gray")
    plt.show()  # waits for window to be closed
    plt.figure(figsize=(1, 1))
    plt.imshow(image[2].squeeze(), cmap="gray")
    plt.show()  # waits for window to be closed
    plt.figure(figsize=(1, 1))
    plt.imshow(image[3].squeeze(), cmap="gray")
    plt.show()  # waits for window to be closed
