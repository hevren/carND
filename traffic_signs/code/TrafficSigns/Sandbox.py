import pickle
from sklearn.model_selection import train_test_split
import random
import csv
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from Helper2 import LeNet

EPOCHS = 20
BATCH_SIZE = 32
rate = 0.0001

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

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(3, 3))
plt.imshow(image, cmap="gray")
print(str(y_train[index]) + '  ' + signnames[y_train[index] + 1][1])
plt.show()

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

count = tf.Variable(0, dtype=tf.int32, name='count')
acount_assign = count.assign_add(1)

value = tf.placeholder(tf.float64, 1)
accuracy_value = tf.Variable(0.0, dtype=tf.float64, name='accuracy_value')
accuracy_value_assign=tf.assign(accuracy_value,value[0])

one_hot_y = tf.one_hot(y, 42)

logits = LeNet(x)
sol1 = tf.nn.relu(logits)
sol2 = tf.argmax(sol1, 1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    # saver.restore(sess, tf.train.latest_checkpoint('.'))
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    validation_accuracy = 0.0
    validation_accuracy_old = -1.0
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)

        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        if validation_accuracy > validation_accuracy_old:
            validation_accuracy_old = validation_accuracy
            count = sess.run(acount_assign)
            sess.run(accuracy_value_assign, feed_dict={value: [validation_accuracy]})
            saver.save(sess, 'data/lenet.ckpt')
            print("model Saved:" + str(count) + " " + str(accuracy_value.eval()))

            # saver.save(sess, './lenet')
            # print("Model saved")
