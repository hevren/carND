import pickle
from sklearn.model_selection import train_test_split
import random
import csv
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from Helper2 import LeNet
import cv2
import numpy as np

#X_train, y_train = shuffle(X_train, y_train)

count = tf.Variable(0, dtype=tf.int32, name='count')
acount_assign = count.assign_add(1)

value = tf.placeholder(tf.float64, 1)
accuracy_value = tf.Variable(0.0, dtype=tf.float64, name='accuracy_value')
accuracy_value_assign=tf.assign(accuracy_value,value[0])

saver = tf.train.Saver()

with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #count = sess.run(acount_assign)
    #count = sess.run(acount_assign)
    #count = sess.run(acount_assign)
    #sess.run(accuracy_value_assign, feed_dict={value: [99]})
    #saver.save(sess, 'data_test/lenet.ckpt')
    #print("model Saved:" + str(count) + " " + str(accuracy_value.eval()))

    #saver.restore(sess, 'data_test/lenet.ckpt')
    saver.restore(sess, 'data_3_945/lenet.ckpt')
    print("count:" + str(sess.run(count)) + " validation_accuracy:" + str(sess.run(accuracy_value)))

