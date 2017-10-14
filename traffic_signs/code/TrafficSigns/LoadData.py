# Load pickled data
import pickle
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

def load():
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
