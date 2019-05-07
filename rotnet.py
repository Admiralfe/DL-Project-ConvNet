import tensorflow as tf
import rotation

""" Builds an MLPBlock of a neural network.

	input - image data input to the block
	conv1_shape - shape of the first convolutional filter in the blcok
	l2_channels - the number of output channels from the second layer
	out_channels - the number of output channels from the entire block
	max_pool_size - size of the maxpooling window.
"""
def MLPBlock(input, conv1_shape, l2_channels, out_channels, pool_size, is_final):
	W1 = tf.get_variable("W1", conv1_shape, initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float32)
	b1 = tf.get_variable("b1", [conv1_shape[3]], initializer=tf.random_normal_initializer(stddev=0.01), dtype=tf.float32)
	L1 = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
	L1 = tf.nn.relu(L1)
		
	#The number of input channels here is the same as the number of output channels previously, conv1_dim[3].
	W2 = tf.get_variable("W2", [1, 1, conv1_shape[3], l2_channels], initializer=tf.random_normal_initializer(stddev=0.01))
	b2 = tf.get_variable("b2", [l2_channels], initializer=tf.constant_initializer(0), dtype=tf.float32)
	L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
	L2 = tf.nn.relu(L2)

	W3 = tf.get_variable("W3", [1, 1, l2_channels, out_channels], initializer=tf.random_normal_initializer(stddev=0.01))
	b3 = tf.get_variable("b3", [out_channels], initializer=tf.constant_initializer(0), dtype=tf.float32)
	L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
	L3 = tf.nn.relu(L3)
	
	if is_final:
		L3 = tf.nn.avg_pool(L3, ksize=[1, pool_size, pool_size, 1], strides=[1, 1, 1, 1], padding="VALID")
	else:
		L3 = tf.nn.max_pool(L3, ksize=[1, pool_size, pool_size, 1], strides=[1, 2, 2, 1], padding="SAME")
	return L3
	
""" Builds the full rotnet, based on a network in network architecture
	
	Returns:
	x - original input matrix
	probs - probability matrix for each of the input vectors and each of the classes
	logits - matrix of unnormalized probabilities
"""
def rotnet():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	_NUM_CLASSES = 4
	
	x = tf.placeholder(shape=(None, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS), dtype=tf.float32)
	y = tf.placeholder(shape=(None, _NUM_CLASSES), dtype=tf.float32)
	x_batch = tf.reshape(x, shape=(-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS))
	
	with tf.variable_scope("MLP_1"):
		output = MLPBlock(x_batch, conv1_shape=[5, 5, 3, 192], l2_channels=160, out_channels=96, pool_size=3, is_final=False)
	
	with tf.variable_scope("MLP_2"):
		output = MLPBlock(output, conv1_shape=[5, 5, 96, 192], l2_channels=192, out_channels=192, pool_size=3, is_final=False)
	
	with tf.variable_scope("MLP_3"):
		output = MLPBlock(output, conv1_shape=[3, 3, 192, 192], l2_channels=192, out_channels=_NUM_CLASSES, pool_size=8, is_final=True)
	
	logits = tf.reshape(output, (-1, _NUM_CLASSES))
	probs = tf.nn.softmax(logits)
	
	return x, y, probs, logits

def train(train_x, train_y, logits):
	_BATCH_SIZE = 128
	#Use a placeholder here to be able to set the learning rate adaptively.
	learning_rate = tf.placeholder(tf.float32, shape=[])
	momentum = 0.9
	weight_decay = 0.0005
	num_epochs = 10
	#L2 loss term for weight decay regularization
	L2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(train_y, logits))
	optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss + L2 * weight_decay)
	
	with tf.Session() as sess:
		for i in range(num_epochs):
			print("Starting epoch %d", i)
			curr_lr = 0.1
			#Define the number of steps in an epoch
			if (len(train_x) % _BATCH_SIZE == 0):
				num_iters = len(train_x) / _BATCH_SIZE
			else:
				num_iters = int(len(train_x) / _BATCH_SIZE) + 1
			
			#Shuffle by permuting indexes and indexing the array. Supposedly this is faster than just using numpy.random.shuffle.
			train_x = train_x[np.random.permutation(x.shape[0])]
			for j in range(num_iters):
				"""Create batches here"""
				sess.run(optimizer, feed_dict={x : x_batch, y : y_batch, learning_rate : curr_lr})