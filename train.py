import math

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import rotnet
import rotation


#Project wide constants and flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("training_steps", 1000,
                            """Number of steps to run training for""")
                            
#Global training constants
NUM_EPOCHS = 1
VALIDATION_BATCH_SIZE = 100
NUM_TRAIN_BATCHES_FOR_LOGGING = 100

def _log_scalar(value, tag, step, summary_writer):
    """ Helper function to log a scalar value for visualization in tensorboard
        
        Args:
            value - python scalar to log
            step - current training step
            tag - name of the logged scalar
            summary_writer - tf.summary.FileWriter to write the log
    """
    
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, step)
    
    return
def train():
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        
        training_images, training_labels = rotation.load_training_data()
        validation_images, validation_labels = rotation.load_validation_data()
        
        training_dataset = rotation.make_tf_dataset(training_images.shape, training_labels.shape)
        validation_dataset = rotation.make_tf_dataset(validation_images.shape, validation_labels.shape)

        #Normalize the images to have zero mean and unit stddev. 
        #Leaves the labels as they are.
        def normalize(img, label):
            return tf.cast(tf.image.per_image_standardization(img), tf.float16), label
        training_dataset = training_dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        #Batch the validation data so we can run through it more quickly.
        validation_dataset = validation_dataset.repeat().batch(VALIDATION_BATCH_SIZE)
        num_validation_iters = math.ceil(len(validation_labels) / VALIDATION_BATCH_SIZE)
        
        num_training_iters = math.ceil(len(training_labels) / FLAGS.batch_size)
        
        #Get the placeholders for iterator creation.
        iter_input_images_train = tf.get_collection("iterator_inputs")[0]
        iter_input_labels_train = tf.get_collection("iterator_inputs")[1]
        iter_input_images_val = tf.get_collection("iterator_inputs")[2]
        iter_input_labels_val = tf.get_collection("iterator_inputs")[3]

        training_iterator = rotation.data_pipeline(training_dataset)
        
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, 
                                                       validation_dataset.output_types, 
                                                       validation_dataset.output_shapes)
        
        images, labels = iterator.get_next()

        #batch the validation data so that we can run through it more quickly when evaluating the model.
        validation_iterator = validation_dataset.make_initializable_iterator()
        
        logits = rotnet.rotnet(images)
        loss = rotnet.loss(logits, labels)
        
        train_op = rotnet.train_op(loss, global_step)
        
        #Get the accuracy node in the graph to use for computing validation accuracy
        accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
        
        #Create a file writer to log progress
        summary_writer = tf.summary.FileWriter("tmp/6")
        
        with tf.Session() as sess:
            #Initialize global variables and iterators
            sess.run(tf.global_variables_initializer())
            
            sess.run(training_iterator.initializer, 
                     feed_dict={iter_input_images_train : training_images, 
                                iter_input_labels_train : training_labels})
            
            sess.run(validation_iterator.initializer, 
                     feed_dict={iter_input_images_val : validation_images, 
                                iter_input_labels_val : validation_labels})
            
            #Feed these handles through "feed dict" when running ops to determine which
            #dataset to use samples from.
            train_handle = sess.run(training_iterator.string_handle())
            val_handle = sess.run(validation_iterator.string_handle())
            summary_writer.add_graph(sess.graph)
            for i in range(FLAGS.training_steps):
                print("starting iteration " + str(i) + " of " + str(FLAGS.training_steps))
                
                sess.run(train_op, feed_dict={handle : train_handle})

                if (i % 100 == 0):
                    summary = sess.run(tf.summary.merge_all(), feed_dict={handle : train_handle})
                    summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    
                    val_loss = 0
                    val_acc = 0
                    #Compute the training and validation losses and accuracies by iterating over the validation set
                    #in batches and summing the result
                    #Log the values to view in Tensorboard later.
                    for _ in range(num_validation_iters):
                        val_loss += sess.run(loss, feed_dict={handle : val_handle}) / num_validation_iters
                        val_acc += sess.run(accuracy, feed_dict={handle : val_handle}) / num_validation_iters

                    #Log the values for viewing in tensorboard later.
                    _log_scalar(val_loss, "validation loss", i, summary_writer)
                    _log_scalar(val_acc, "validation accuracy", i, summary_writer)
                                        
if __name__ == "__main__":          
    train()