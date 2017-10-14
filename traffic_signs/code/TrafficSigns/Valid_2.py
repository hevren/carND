import random
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from Helper2 import LeNet
import cv2
import numpy as np
from os import walk


german_signs=[]
german_signs_id=[]

with open('signnames.csv', 'r') as csvfile:
    signreader = csv.reader(csvfile, delimiter=',')
    signnames = list(signreader)


for (dirpath, dirnames, fname) in walk('./GTS'):
    id=random.randint(1,42)
    index=random.randint(1,10)
    img=cv2.imread('./GTS'+fname)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    german_signs.append(cv2.resize(img,(32,32)))
    german_signs_id.append(id)

fig=plt.figure(figsize=(20, 20))
for i in range(0,5):
    image = german_signs[i].squeeze()
    a=fig.add_subplot(1,6,i+1)
    plt.imshow(image, cmap="gray")
    a.set_title(str(german_signs_id[i]) + '  ' + signnames[german_signs_id[i] + 1][1])

plt.tight_layout()
print("image shape:"+str(np.shape(german_signs[0])))

plt.show()

count = tf.Variable(0, dtype=tf.int32, name='count')
accuracy_value = tf.Variable(0.0, dtype=tf.float64, name='accuracy_value')
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 42)

logits = LeNet(x)
sol1 = tf.nn.relu(logits)
sol2 = tf.argmax(sol1, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'data_3_945/lenet.ckpt')
    print("count:" + str(sess.run(count)) + " validation_accuracy:" + str(sess.run(accuracy_value)))

    num_examples = len(german_signs)

    res = sess.run(sol2, feed_dict={x: german_signs})
    print("prediction:" + str(res))
    print("prediction:" + str(signnames[res[0] + 1]) + " " + str(signnames[res[1] + 1]) + " " +
          str(signnames[res[2] + 1]) + " " + str(signnames[res[3] + 1])+ " " + str(signnames[res[4] + 1]))
    print("actual:" + str(signnames[german_signs_id[0] + 1]) + " " + str(signnames[german_signs_id[1] + 1]) + " " +
          str(signnames[german_signs_id[2] + 1]) + " " + str(signnames[german_signs_id[3] + 1])+ " " + str(signnames[german_signs_id[4] + 1]))

    fig = plt.figure(figsize=(20, 20))
    for i in range(0, 5):
        image = german_signs[i].squeeze()
        a = fig.add_subplot(1, 6, i + 1)
        plt.imshow(image, cmap="gray")
        a.set_title(str(german_signs_id[i]) + '  ' + signnames[german_signs_id[i] + 1][1])

    plt.tight_layout()
    plt.show()  # waits for window to be closed
