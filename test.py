import tensorflow as tf
import numpy as np
from random import randint

dims = 2
#pos = randint(0, dims - 1)
pos = 1

#logits = tf.random_uniform([dims], maxval=1, dtype=tf.float32)
logits = tf.constant([0.0, 1.0])
labels = tf.one_hot(pos, dims)
output = tf.constant([-100.0, 100.0])

print 'logits: ', logits
print 'labels: ', labels

res1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(pos))
res3 = tf.nn.softmax(logits)

with tf.Session() as sess:
    a, b = sess.run([res1, res2])
    print a, b
    print a == b

    soft_logits = res3.eval()
    cross_entropy = -1.0 * np.log(soft_logits[1])
    print 'Softmax logits: ', soft_logits
    print 'Cross entropy: ', cross_entropy
    
