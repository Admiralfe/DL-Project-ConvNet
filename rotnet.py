import tensorflow as tf
import numpy as np
import data
import math

"""Project wide flags"""
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("batch_size", 128,
                            """Batch size to use for training""")
                            
tf.app.flags.DEFINE_integer("batch_size_final", 128,
                            """Batch size to use when training the final classifier""")

tf.app.flags.DEFINE_boolean("use_fp16", True,
                            """Use 16 bit floating point""")
                    

"""Global constants"""
#Image set constants
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
NUM_CLASSES_ROTATION = 4
NUM_CLASSES_CIFAR = 10
NUM_TRAINING_SAMPLES = 160000

#Training constants
WEIGHT_DECAY = 0.001
DECAY_STEPS = 3000
INITIAL_LR = 0.1
DECAY_RATE = 0.2
MOMENTUM = 0.9
NESTEROV = True

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
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(
        name,
        shape, 
        initializer=tf.random_normal_initializer(0, tf.cast(stddev, dtype), dtype=dtype),
        dtype=dtype)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), tf.cast(tf.constant(wd), dtype), name="weight_loss")
    tf.add_to_collection("losses", weight_decay)
    return var


def MLPBlock(input, conv1_shape, l2_channels, out_channels, renorm):
    """ Builds an MLPBlock of a neural network.
    Args:
        input - image data input to the block
        conv1_shape - shape of the first convolutional filter in the blcok
        l2_channels - the number of output channels from the second layer
        out_channels - the number of output channels from the entire block
    returns:
        Output from the MLPBlock.
    """
    
    #Placeholder flag for whether the model is currently used for inference or training.
    #This is needed to perform batch normalization correctly.
    training_flag = tf.get_default_graph().get_tensor_by_name("training_flag:0")
    W1 = _create_variable("W1", 
                          shape=conv1_shape, 
                          stddev=tf.constant(math.sqrt(2.0 / (conv1_shape[0] * conv1_shape[1] * conv1_shape[3]))), 
                          wd=WEIGHT_DECAY)
    tf.summary.histogram("Weight_1", W1)
    L1 = tf.nn.conv2d(input, W1, name="L1", strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.layers.batch_normalization(L1, momentum=0.9, epsilon=0.00001, training=training_flag, renorm=renorm)
    L1 = tf.nn.relu(L1)
    tf.summary.histogram("block_1", L1)
        
    #The number of input channels here is the same as the number of output channels previously, conv1_dim[3].
    W2 = _create_variable("W2", 
                          shape=[1, 1, conv1_shape[3], l2_channels], 
                          stddev=tf.constant(math.sqrt(2 / l2_channels)), 
                          wd=WEIGHT_DECAY)
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.layers.batch_normalization(L2, momentum=0.9, epsilon=0.00001, training=training_flag, renorm=renorm)
    L2 = tf.nn.relu(L2)
    tf.summary.histogram("block_2", L2)

    W3 = _create_variable("W3", 
                          [1, 1, l2_channels, out_channels], 
                          stddev=tf.constant(math.sqrt(2 / out_channels)), 
                          wd=WEIGHT_DECAY)
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.layers.batch_normalization(L3, momentum=0.9, epsilon=0.00001, training=training_flag, renorm=renorm)
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
    
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    
    tf.summary.image("images", x_batch, 4)
    #Flag for whether the model is currently used for training or inference
    #This is needed to perform the batch normalization correctly.
    training = tf.placeholder(tf.bool, name="training_flag")
    
    with tf.variable_scope("MLP_1"):
        output = MLPBlock(x_batch, conv1_shape=[5, 5, 3, 192], l2_channels=160, out_channels=96)
        output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        tf.summary.histogram("pool_out", output)
    
    with tf.variable_scope("MLP_2"):
        output = MLPBlock(output, conv1_shape=[5, 5, 96, 192], l2_channels=192, out_channels=192)
        output = tf.nn.avg_pool(output, name="output", ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    with tf.variable_scope("MLP_3"):
        output = MLPBlock(output, conv1_shape=[3, 3, 192, 192], l2_channels=192, out_channels=192)
        output = tf.reduce_mean(output, axis=[1, 2])
        """output = tf.nn.avg_pool(output, 
                                ksize=[1,output.shape[1], output.shape[2], 1], 
                                strides=[1, output.shape[1], output.shape[2], 1],
                                padding="VALID")
        """
        tf.summary.histogram("pool_out", output)
        
    with tf.variable_scope("Linear_layer"):
        flattened = tf.reshape(output, (-1, 192))
        W = tf.get_variable("W", 
                             shape=[192, NUM_CLASSES_ROTATION],
                             initializer=tf.random_uniform_initializer(tf.cast(tf.constant(-1 * math.sqrt(1 / 192)), dtype), 
                                                                       tf.cast(tf.constant(math.sqrt(1 / 192)), dtype)),
                             dtype=dtype)
        tf.summary.histogram("W", W)
        weight_decay_W = tf.multiply(tf.nn.l2_loss(W), tf.cast(tf.constant(WEIGHT_DECAY), dtype), name="weight_loss")
        tf.add_to_collection("losses", weight_decay_W)
        b = tf.get_variable("b",
                            shape=[NUM_CLASSES_ROTATION],
                            initializer=tf.constant_initializer(0),
                            dtype=dtype)
        tf.summary.histogram("b", b)
        logits = tf.nn.xw_plus_b(flattened, W, b)
        tf.summary.histogram("linear_layer", logits)

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

    labels = tf.cast(labels, tf.int32)
    sample_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="cross_entropy_per_sample")
    cross_entropy = tf.reduce_mean(sample_cross_entropy, name="cross_entropy")
    
    tf.add_to_collection("losses", cross_entropy)
    correct_predictions = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
    tf.summary.scalar("Training accuracy", accuracy)
    
    #The total loss is the cross entropy loss plus 
    #the weight decay losses from the variables
    return tf.add_n(tf.get_collection("losses"), name="total_loss")
    
def train_op(total_loss, global_step):
    """Creates a training operation
    
        Apply one step of momentum gradient descent to all trainable
        variables. Compute moving averages of total loss and add to summary.
        
        Args:
            total_loss - loss returned by loss()
            global_step - global_step in the training
    """
    loss_avg = tf.train.ExponentialMovingAverage(0.9)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        loss_avg_op = loss_avg.apply([total_loss])
    tf.summary.scalar("total_loss", loss_avg.average(total_loss))
    
    #TODO: implement the correct dropping of learning rates.
    batches_per_epoch = NUM_TRAINING_SAMPLES / FLAGS.batch_size
    lr = tf.train.exponential_decay(
            INITIAL_LR,
            global_step,
            DECAY_STEPS,
            DECAY_RATE,
            staircase=True)
    tf.summary.scalar("Learning rate", lr)
    
    with tf.control_dependencies([loss_avg_op]):
        grad_opt = tf.train.MomentumOptimizer(lr, MOMENTUM, use_nesterov=NESTEROV).minimize(total_loss, global_step=global_step)
    
    #POTENTIAL TODO: add more summaries for gradients etc.
    return grad_opt

def make_final_classifier(checkpoint_dir):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    
    rotnet_saver = tf.train.import_meta_graph(checkpoint_dir + "/feature_model.meta")
    rotnet_graph = tf.get_default_graph()
    
    pretrained_output = rotnet_graph.get_tensor_by_name("MLP_2/output:0")
    #Flag for whether the model is currently used for training or inference
    #This is needed to perform the batch normalization correctly.
    training = tf.placeholder(tf.bool, name="training_flag")
    
    with tf.variable_scope("MLP_3_classifier"):
        #Freeze the old gradients so that the pretrained features remain constant
        output = tf.stop_gradient(pretrained_output)
        output = MLPBlock(output, conv1_shape=[3, 3, 192, 192], l2_channels=192, out_channels=192, renorm=True)
        output = tf.reduce_mean(output, axis=[1,2])
        flattened = tf.reshape(output, (-1, 192))
        W = tf.get_variable("W", 
                         shape=[192, NUM_CLASSES_CIFAR],
                         initializer=tf.random_uniform_initializer(tf.cast(tf.constant(-1 * math.sqrt(1 / 192)), dtype), 
                                                                   tf.cast(tf.constant(math.sqrt(1 / 192)), dtype)),
                         dtype=dtype)
        tf.summary.histogram("W", W)
        weight_decay_W = tf.multiply(tf.nn.l2_loss(W), tf.cast(tf.constant(WEIGHT_DECAY), dtype), name="weight_loss")
        tf.add_to_collection("cifar_10_losses", weight_decay_W)
        b = tf.get_variable("b",
                            shape=[NUM_CLASSES_CIFAR],
                            initializer=tf.constant_initializer(0),
                            dtype=dtype)
        tf.summary.histogram("b", b)
        logits = tf.nn.xw_plus_b(flattened, W, b)
        tf.summary.histogram("Logits", logits)
            
    return logits
if __name__ == "__main__":
    make_final_classifier()
    
    
    