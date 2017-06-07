import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

Y = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1, 784]), w) +b)
Y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

optimizer = tf.train.GradientDescentOptimizer(0.002)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={x: batch_X, Y: batch_Y}

    sess.run(train_step, feed_dict=train_data)

    
