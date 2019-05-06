import tensorflow as tf
import numpy as np
import rotnet
import rotation
import os


#Test if the forward pass of the graph works.
def test_graph():
	train_data, train_labels, val_data, val_labels, test_data, test_labels = rotation.load_data()
	test_image = np.reshape(train_data[0], (1, 3, 32, 32)).transpose([0, 2, 3, 1])
	x, probs, logits = rotnet.rotnet()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		probs, logits = sess.run([probs, logits], feed_dict={x: test_image})
		print(probs)
	return

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_graph()