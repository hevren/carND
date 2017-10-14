import tensorflow as tf

value = tf.placeholder(tf.float32, 1)

count = tf.Variable(0)
assign = count.assign(1)

accuracy_value = tf.Variable(0)

valueTF = tf.Variable(0,dtype=tf.float32)
assign_value = tf.assign(valueTF, value[0])

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(assign)
    sess.run(accuracy_value, feed_dict={accuracy_value: 2})
    sess.run(assign_value, feed_dict={value: [3]})

    print(str(count.eval()) + ", " + str(accuracy_value.eval()) + ", " + str(valueTF.eval()))

    saver.save(sess, 'data/test1.ckpt')
