import tensorflow as tf

count = tf.Variable(0, dtype=tf.int32)
accuracy_value = tf.Variable(0.0)
valueTF = tf.Variable(0,dtype=tf.float32)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'data/test1.ckpt')
    c1 = sess.run(count)
    c2 = sess.run(accuracy_value)
    c3 = sess.run(valueTF)

print(str(c1) + ", " + str(c2) + ", " + str(c3))
