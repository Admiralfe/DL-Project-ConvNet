import tensorflow as tf
import numpy as np
import math

import rotnet
import rotation


#Project wide constants and flags

FLAGS = tf.app.flags.FLAGS

#Global training constants
NUM_EPOCHS = 1


def train():
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()
		
		training_images, training_labels = rotation.load_training_data()
		num_training_samples = len(training_labels)
		iters_per_epoch = math.ceil(num_training_samples / FLAGS.batch_size)
		
		dataset = rotation.make_tf_dataset(images_shape=training_images.shape, labels_shape=training_labels.shape)
		"""
		def map_fun(img, label):
			return tf.cast(tf.image.per_image_standardization(img), tf.float16), label
		dataset = dataset.map(map_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		"""
		images, labels, iterator_initializer = rotation.data_pipeline(dataset, num_epochs=NUM_EPOCHS)
		
		logits = rotnet.rotnet(images)
		loss = rotnet.loss(logits, labels)
		
		summary_writer = tf.summary.FileWriter("tmp/2")
		
		"""Test code pls ignore
		n = tf.data.Dataset.range(8).batch(4).shuffle(buffer_size=20).repeat(5)
		n = n.apply(tf.data.experimental.unbatch())
		iter = n.batch(8).make_one_shot_iterator().get_next()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print([sess.run(iter) for _ in range(5)])
		exit()
		"""
		
		train_op = rotnet.train_op(loss, global_step)
		iter_input_images = tf.get_collection("iterator_inputs")[0]
		iter_input_labels = tf.get_collection("iterator_inputs")[1]
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(iterator_initializer, 
					 feed_dict={iter_input_images : training_images, 
								iter_input_labels : training_labels})
								
			for i in range(NUM_EPOCHS):
				print("Starting epoch ", i)
				for j in range(iters_per_epoch):
					if (tf.train.global_step(sess, global_step) % 5 == 0 and tf.train.global_step(sess, global_step) > 0):
						summary = sess.run(tf.summary.merge_all())
						summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
					print("Starting iter " + str(j) + " of " + str(iters_per_epoch))
					sess.run(train_op)
					
					
train()