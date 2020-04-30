import tensorflow as tf
import random
import time
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)
#one_hot = True, 4-> 0, 0, 0, 0, 1, 0, 0, 0, 0, 0

X = tf.placeholder(tf.float32, [None, 784], name="X")
Y = tf.placeholder(tf.float32, [None, 10], name="Y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # 살릴확률

#W1 = tf.Variable(tf.random_normal([784, 600]))
W1 = tf.get_variable("W1", shape=[784, 600], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([600]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[600, 600], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([600]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[600, 800], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([800]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[800, 10], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.xw_plus_b(L3, W4, b4, name="hypothesis") #L3W4 + b4
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)) # Y = [0,0,0,1,0,0,0,0,0,0]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summary_op = tf.summary.scalar("accuracy", accuracy)
# define cost/loss & optimizer

# hypothesis = tf.nn.softmax(hypothesis)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))


learning_rate = 0.001
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #back propagation

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 30
batch_size = 100

# ========================================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
val_summary_dir = os.path.join(out_dir, "summaries", "dev")
val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
# ========================================================================

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

max = 0
dropout = 0.8
early_stopped = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) #iteration 55000/ 100 = 550

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: dropout}
        c, _, a = sess.run([cost, optimizer, summary_op], feed_dict=feed_dict)
        avg_cost += c / total_batch

    #......
    print('Epoch:', '%04d' % (epoch + 1), 'training cost =', '{:.9f}'.format(avg_cost))
    # ========================================================================
    train_summary_writer.add_summary(a, early_stopped) ##
    val_accuracy, summaries = sess.run([accuracy, summary_op], feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, keep_prob: 1.0})
    val_summary_writer.add_summary(summaries, early_stopped) ##
    # ========================================================================
    print('Validation Accuracy:', val_accuracy)
    if val_accuracy > max:
        max = val_accuracy
        early_stopped = epoch + 1
        saver.save(sess, checkpoint_prefix, global_step=early_stopped)
#......
print('Learning Finished!')
print('Validation Max Accuracy:', max)
print('Early stopped time:', early_stopped)

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
print('Test Latest Accuracy:', test_accuracy)