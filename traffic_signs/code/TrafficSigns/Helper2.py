import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x12.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 12), mean=mu, stddev=sigma), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(12), name='conv1_b')
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x12. Output = 14x14x12.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x24.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 24), mean=mu, stddev=sigma), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(24), name='conv2_b')
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x24.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x24. Output = 600.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 600. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(600, 400), mean=mu, stddev=sigma), name='fc1_W')
    fc1_b = tf.Variable(tf.zeros(400), name='fc1_b')
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # # SOLUTION: Layer 4: Fully Connected. Input = 400. Output = 300.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(400, 300), mean=mu, stddev=sigma), name='fc2_W')
    fc2_b = tf.Variable(tf.zeros(300), name='fc2_b')
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # # SOLUTION: Layer 5: Fully Connected. Input = 300. Output = 120.
    fc21_W = tf.Variable(tf.truncated_normal(shape=(300, 120), mean=mu, stddev=sigma), name='fc21_W')
    fc21_b = tf.Variable(tf.zeros(120), name='fc21_b')
    fc21 = tf.matmul(fc2, fc21_W) + fc21_b

    # SOLUTION: Activation.
    fc21 = tf.nn.relu(fc21)

    # SOLUTION: Layer 6: Fully Connected. Input = 120. Output = 42.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(120, 42), mean=mu, stddev=sigma), name='fc3_W')
    fc3_b = tf.Variable(tf.zeros(42), name='fc3_b')
    logits = tf.matmul(fc21, fc3_W) + fc3_b

    return logits