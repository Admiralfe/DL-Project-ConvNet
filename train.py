import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math

import rotnet
import rotation


#Project wide constants and flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("training_steps", 100,
							"""Number of steps to run training for""")
#Global training constants
NUM_EPOCHS = 1


def train():
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()
		
		training_images, training_labels = rotation.load_training_data()
		num_training_samples = len(training_labels)
		iters_per_epoch = math.ceil(num_training_samples / FLAGS.batch_size)
		
		dataset = rotation.make_tf_dataset(num_training_samples)

		def map_fun(img, label):
			return tf.cast(tf.image.per_image_standardization(img), tf.float16), label
		dataset = dataset.map(map_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)

		#Create a file writer to log progress
		summary_writer = tf.summary.FileWriter("tmp/5")
		#Get the placeholders for iterator creation.
		iter_input_images = tf.get_collection("iterator_inputs")[0]
		iter_input_labels = tf.get_collection("iterator_inputs")[1]

		images, labels, iterator_initializer = rotation.data_pipeline(dataset)
		
		logits = rotnet.rotnet(images)
		loss = rotnet.loss(logits, labels)
		
		train_op = rotnet.train_op(loss, global_step)
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(iterator_initializer, 
					 feed_dict={iter_input_images : training_images, 
								iter_input_labels : training_labels})
			summary_writer.add_graph(sess.graph)
			for i in range(FLAGS.training_steps):
				print("starting iteration " + str(i) + " of " + str(FLAGS.training_steps))
				if (i % 10 == 0):
					summary = sess.run(tf.summary.merge_all())
					summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
				sess.run(train_op)
					
					
train()