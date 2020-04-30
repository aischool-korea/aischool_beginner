import tensorflow as tf

x_data = [[1, 2]]

X = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.nn.sigmoid(tf.matmul(X, W) + b)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    prediction = sess.run(hypothesis, feed_dict={X: x_data})
    print(prediction)


