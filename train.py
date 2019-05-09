import tensorflow as tf
import numpy as np
import math

import rotnet
import rotation


#Project wide constants and flags

FLAGS = tf.app.flags.FLAGS

#Global training constants
NUM_EPOCHS = 100


def train():
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()
		
		training_images, training_labels = rotation.load_training_data()
		num_training_samples = len(training_labels)
		iters_per_epoch = math.ceil(num_training_samples / FLAGS.batch_size)
		
		logits = rotnet.rotnet()
		loss = rotnet.loss(logits)
		
		train_op = rotnet.train_op(loss, global_step)
		dataset = rotation.make_tf_dataset(images_shape=training_images.shape, labels_shape=training_labels.shape)
		iter_input_images = tf.get_collection("iterator_inputs")[0]
		iter_input_labels = tf.get_collection("iterator_inputs")[1]
		graph_input_images = tf.get_collection("inputs")[0]
		graph_input_labels = tf.get_collection("inputs")[1]
		
		with tf.Session() as sess:
			for i in range(NUM_EPOCHS):
				print("Starting epoch ", i)
				iterator = rotation.make_epoch_iterator(dataset)
				sess.run(iterator.initializer, feed_dict={iter_input_images : training_images, iter_input_labels : training_labels})
				for j in range(iters_per_epoch):
					print("Starting iter " + str(j) + " of " + str(iters_per_epoch))
					batch_images, batch_labels = iterator.get_next()
					print(batch_images.shape)
					sess.run(train_op, feed_dict={graph_input_images : batch_images, graph_input_labels : batch_labels})
					
					
train()