import tensorflow as tf
import numpy as np
import rotation
import math

"""Project wide flags"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("batch_size", 4 * 128,
							"""Batch size to use for training""")

tf.app.flags.DEFINE_boolean("use_fp16", True,
							"""Use 16 bit floating point""")


"""Global constants"""
#Image set constants
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
NUM_CLASSES = 4
NUM_TRAINING_SAMPLES = 160000

#Training constants
WEIGHT_DECAY = 0.0005
EPOCHS_PER_DECAY = 30
INITIAL_LR = 0.1
DECAY_RATE = 0.2
MOMENTUM = 0.9

def _create_variable(name, shape, stddev, wd):
	"""Helper function to create variables with weight_decay
	
	Args:
		name : name of variable
		shape : list of ints
		stddev : standard deviation to initialize with
		wd : weight decay, for L2 loss, if None, weight decay not used.
	
	Returns:
		Variable tensor
	"""
	var = tf.get_variable(
		name,
		shape, 
		initializer=tf.random_normal_initializer(0, tf.cast(stddev, tf.float16), dtype=tf.float16),
		dtype=tf.float16)
	weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
	tf.add_to_collection("losses", weight_decay)
	return var


def MLPBlock(input, conv1_shape, l2_channels, out_channels):
	""" Builds an MLPBlock of a neural network.
	Args:
		input - image data input to the block
		conv1_shape - shape of the first convolutional filter in the blcok
		l2_channels - the number of output channels from the second layer
		out_channels - the number of output channels from the entire block
	returns:
		Output from the MLPBlock.
	"""

	W1 = _create_variable("W1", 
						  shape=conv1_shape, 
						  stddev=tf.constant(math.sqrt(2.0 / (conv1_shape[0] * conv1_shape[1] * conv1_shape[3]))), 
						  wd=WEIGHT_DECAY)
	tf.summary.histogram("Weight_1", W1)
	#b1 = tf.get_variable("b1", shape=[conv1_shape[3]], initializer=tf.constant_initializer(0), dtype=tf.float16)
	L1 = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
	L1 = tf.layers.batch_normalization(L1, momentum=0.9, epsilon=0.00001)
	L1 = tf.nn.relu(L1)
	tf.summary.histogram("block_1", L1)
		
	#The number of input channels here is the same as the number of output channels previously, conv1_dim[3].
	W2 = _create_variable("W2", 
						  shape=[1, 1, conv1_shape[3], l2_channels], 
						  stddev=tf.constant(math.sqrt(2 / l2_channels)), 
						  wd=WEIGHT_DECAY)
	#b2 = tf.get_variable("b2", [l2_channels], initializer=tf.constant_initializer(0), dtype=tf.float16)
	L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
	L2 = tf.layers.batch_normalization(L2, momentum=0.9, epsilon=0.00001)
	L2 = tf.nn.relu(L2)
	tf.summary.histogram("block_2", L2)

	W3 = _create_variable("W3", 
						  [1, 1, l2_channels, out_channels], 
						  stddev=tf.constant(math.sqrt(2 / out_channels)), 
						  wd=WEIGHT_DECAY)
	#b3 = tf.get_variable("b3", [out_channels], initializer=tf.constant_initializer(0), dtype=tf.float16)
	L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
	L3 = tf.layers.batch_normalization(L3, momentum=0.9, epsilon=0.00001)
	L3 = tf.nn.relu(L3)
	tf.summary.histogram("block_3", L3)
	"""
	if is_final:
		L3 = tf.nn.avg_pool(L3, ksize=[1, pool_size, pool_size, 1], strides=[1, 1, 1, 1], padding="VALID")
	else:
		L3 = tf.nn.max_pool(L3, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding="SAME")
	"""
	return L3
	

def rotnet(x_batch):
	""" Builds the full rotnet, based on a network in network architecture
		Args:
			x_batch - input batch of images
		Returns:
			logits - matrix of unnormalized probabilities
	"""
	
	#x = tf.placeholder(name="x_input", shape=(None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), dtype=tf.float16)
	#tf.add_to_collection("inputs", x)
	#x_batch = tf.reshape(x, shape=(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
	
	with tf.variable_scope("MLP_1"):
		output = MLPBlock(x_batch, conv1_shape=[5, 5, 3, 192], l2_channels=160, out_channels=96)
		output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
		tf.summary.histogram("pool_out", output)
	
	with tf.variable_scope("MLP_2"):
		output = MLPBlock(output, conv1_shape=[5, 5, 96, 192], l2_channels=192, out_channels=192)
		output = tf.nn.avg_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
	
	with tf.variable_scope("MLP_3"):
		output = MLPBlock(output, conv1_shape=[3, 3, 192, 192], l2_channels=192, out_channels=192)
		output = tf.nn.avg_pool(output, 
								ksize=[1,output.shape[1], output.shape[2], 1], 
								strides=[1, output.shape[1], output.shape[2], 1],
								padding="VALID")
		tf.summary.histogram("pool_out", output)
		
	with tf.variable_scope("Linear_layer"):
		flattened = tf.reshape(output, (-1, 192))
		W = tf.get_variable("W", 
							 shape=[192, NUM_CLASSES],
							 initializer=tf.random_uniform_initializer(-0.5, 0.5),
							 dtype=tf.float16)
		tf.summary.histogram("W", W)
		b = tf.get_variable("b",
							shape=[NUM_CLASSES],
							initializer=tf.constant_initializer(0),
							dtype=tf.float16)
		logits = tf.nn.xw_plus_b(flattened, W, b)
		tf.summary.histogram("linear_layer", logits)
					
	#logits = tf.reshape(output, (-1, NUM_CLASSES))

	return logits
	
def loss(logits, labels):
	"""
		Add L2 loss for all the trainable variables.
		Adds a summary for the loss.
		
		Args:
			logits - logits from rotnet()
			labels - labels for the batch used to compute logits.
		
		Returns:
			tf tensor with the scalar loss value.
	"""
	#labels = tf.placeholder(name="y_input", shape=(NUM_CLASSES), dtype=tf.uint8)
	tf.add_to_collection("inputs", labels)

	labels = tf.cast(labels, tf.int32)
	sample_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name="cross_entropy_per_sample")
	cross_entropy = tf.reduce_mean(sample_cross_entropy, name="cross_entropy")
	
	tf.add_to_collection("losses", cross_entropy)
	
	#The total loss is the cross entropy loss plus 
	#the weight decay losses from the variables
	return tf.add_n(tf.get_collection("losses"), name="total_loss")
	
def train_op(total_loss, global_step):
	"""Creates a training operation
	
		Apply one step of momentum gradient descent to all trainable
		variables. Compute moving averages of total loss and add to summary.
	"""
	#loss_avg = tf.train.ExponentialMovingAverage(0.9)
	#loss_avg_op = loss_avg.apply([total_loss])
	#tf.summary.scalar("total_loss", loss_avg.average(total_loss))
	tf.summary.scalar("total_loss", total_loss)
	#TODO: implement the correct dropping of learning rates.
	batches_per_epoch = NUM_TRAINING_SAMPLES / FLAGS.batch_size
	decay_steps = math.ceil(batches_per_epoch * EPOCHS_PER_DECAY)
	lr = tf.train.exponential_decay(
			INITIAL_LR,
			global_step,
			decay_steps,
			DECAY_RATE,
			staircase=True)
	
	#with tf.control_dependencies([loss_avg_op]):
	grad_opt = tf.train.MomentumOptimizer(lr, MOMENTUM).minimize(total_loss, global_step=global_step)
	
	#POTENTIAL TODO: add more summaries for gradients etc.
	return grad_opt

def train(train_x, train_y, logits):
	_BATCH_SIZE = 4 * 128
	#Use a placeholder here to be able to set the learning rate adaptively.
	learning_rate = tf.placeholder(tf.float16, shape=[])
	momentum = 0.9
	weight_decay = 0.0005
	num_epochs = 1
	x_input = tf.get_default_graph().get_tensor_by_name("x_input:0")
	y_input = tf.get_default_graph().get_tensor_by_name("y_input:0")
	loss = tf.get_default_graph().get_tensor_by_name("loss:0")
	#L2 loss term for weight decay regularization
	L2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss + L2 * weight_decay)
	
	#Create the one-hot encoding of the labels.
	y_onehot = np.eye(4)[train_y]
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_epochs):
			print("Starting epoch ", i)
			curr_lr = 0.1
			#Define the number of steps in an epoch
			if (len(train_x) % _BATCH_SIZE == 0):
				num_iters = int(len(train_x) / _BATCH_SIZE)
			else:
				num_iters = math.ceil(len(train_x) / _BATCH_SIZE)
			
			#Shuffle the 4 wide blocks of rotated images by splitting the array into blocks of 4, shuffling the blocks
			#And reconnecting them.
			shuffle_blocks = np.split(train_x, 4, axis=0)
			np.random.shuffle(shuffle_blocks)
			train_x = np.concatenate(tuple(shuffle_blocks))
			for j in range(num_iters):
				print("Starting iteration " + str(j) + " of " + str(num_iters))
				"""Create batches here"""
				if (j < num_iters - 1):
					x_batch = train_x[j * _BATCH_SIZE : (j + 1) * _BATCH_SIZE]
					y_batch = y_onehot[j * _BATCH_SIZE : (j + 1) * _BATCH_SIZE]
				else:
					x_batch = train_x[j * _BATCH_SIZE : len(train_x)]
					y_batch = y_onehot[j * _BATCH_SIZE : y_onehot.shape[0]]
				
				sess.run(optimizer, feed_dict={x_input : x_batch, y_input : y_batch, learning_rate : curr_lr})
		
		curr_loss = sess.run(loss, feed_dict={x_input : x_batch, y_input: y_batch})
		print(curr_loss)
	return

def compute_accuracy(data_batch, labels):
	_BATCH_SIZE = 100
	#Get the tensors to feed to the graph
	x_input = tf.get_default_graph().get_tensor_by_name("x_input:0")
	probs = tf.get_default_graph().get_tensor_by_name("probs:0")
	
	if (len(data_batch) % _BATCH_SIZE) == 0:
		num_iters = len(data_batch) / _BATCH_SIZE
	else:
		num_iters = math.ceil(len(data_batch) / _BATCH_SIZE)
	
	probabilities = np.empty(shape=(len(data_batch), 4), dtype=np.float16)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		"""Split the run into multiple chunks to conserve memory"""
		for i in range(num_iters):
			if i < num_iters - 1:
				batch = data_batch[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
				probabilities[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE] = sess.run(probs, feed_dict={x_input : batch})
			else:
				batch = data_batch[i * _BATCH_SIZE : len(data_batch)]
				probabilities[i * _BATCH_SIZE : len(data_batch)] = sess.run(probs, feed_dict={x_input : batch})

	
	guesses = np.argmax(probs, axis=1)
	correct_guesses = 0
	for i in range(guesses.shape[0]):
		if guesses[i] == labels[i]:
			correct_guesses += 1
	
	return correct_guesses / guesses.shape[0]