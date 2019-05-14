import math
import random

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import rotnet
import data

#Project wide constants and flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("training_steps", 1000,
                            """Number of steps to run training for""")
tf.app.flags.DEFINE_string("checkpoint_path", "tmp/checkpoints",
                           """Directory to save model variables to after training""")

#Global training constants
NUM_EPOCHS = 1
VALIDATION_BATCH_SIZE = 25
LOG_INTERVAL = 500

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
    
def eval(checkpoint_dir):
    """ Evaluates the test accuracy of the model
        Args:
            checkpoint_dir - directory where the model to be evaluated is stored
    """
    with tf.Graph().as_default():
        test_images, test_labels = data.load_cifar_test_data()
        
        test_dataset = data.make_tf_dataset(test_images.shape, test_labels.shape)
        test_dataset = test_dataset.map(data.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.prefetch(1)
        test_dataset = test_dataset.map(data.create_rotated_images_with_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
        test_dataset = test_dataset.batch(VALIDATION_BATCH_SIZE)
        
        test_iterator = test_dataset.make_initializable_iterator()
        iter_input_images = tf.get_collection("iterator_inputs")[0]
        iter_input_labels = tf.get_collection("iterator_inputs")[1]
        
        images, labels = test_iterator.get_next()   
        
        logits = rotnet.rotnet(images)
        loss = rotnet.loss(logits, labels)
        
        batches_per_epoch = math.ceil(len(test_labels) / VALIDATION_BATCH_SIZE)
        
        #Get the accuracy node, so we can compute the test accuracy.
        accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
        #Flag needed to say if batch normalization is for inference or training.
        is_training = tf.get_default_graph().get_tensor_by_name("training_flag:0")

        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(test_iterator.initializer, 
                     feed_dict={iter_input_images : test_images, 
                                iter_input_labels : test_labels})
            saver.restore(sess, checkpoint_dir)
            test_accuracy = 0;
            for _ in range(batches_per_epoch):
                test_accuracy += sess.run(accuracy, feed_dict={is_training : False}) / batches_per_epoch
        
        print("The final test accuracy was: ", test_accuracy)
            
def create_rotation_dataset(dataset):
    #Creates a new dataset containing tensors with all four rotated images as one single 4-D tensor
    dataset = dataset.map(data.create_rotated_images_with_labels, 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #Make the dataset into sets of 3-D tensors again, where each tensor is just one copy of rotated images.
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    
    return dataset
    

def train():
    """ Runs the training, logs progress in TensorBoard in certain intervals specified by LOG_INTERVAL
        Also saves the model to disk when training has finished.
    """
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        
        training_images, training_labels = data.load_cifar_training_data()
        validation_images, validation_labels = data.load_cifar_validation_data()
        
        training_dataset = data.make_tf_dataset(training_images.shape, training_labels.shape)

        #Normalize and apply random crop and horizontal flips to the images.
        training_dataset = data.pre_process_data(training_dataset)
        
        training_dataset = training_dataset.shuffle(buffer_size=1000)
        training_dataset = training_dataset.prefetch(1).repeat()
        #Create the rotated data
        training_dataset = create_rotation_dataset(training_dataset)
        training_dataset = training_dataset.batch(FLAGS.batch_size)
        
        #Create the validation set
        validation_dataset = data.make_tf_dataset(validation_images.shape, validation_labels.shape)
        validation_dataset = validation_dataset.map(data.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = create_rotation_dataset(validation_dataset)
        
        #Batch the validation data so we can run through it more quickly.
        validation_dataset = validation_dataset.repeat().prefetch(1).batch(VALIDATION_BATCH_SIZE)
        num_validation_iters = math.ceil(len(validation_labels) / VALIDATION_BATCH_SIZE)
        
        num_training_iters = math.ceil(len(training_labels) / FLAGS.batch_size)
        
        #Get the placeholders for iterator creation.
        iter_input_images_train = tf.get_collection("iterator_inputs")[0]
        iter_input_labels_train = tf.get_collection("iterator_inputs")[1]
        iter_input_images_val = tf.get_collection("iterator_inputs")[2]
        iter_input_labels_val = tf.get_collection("iterator_inputs")[3]
        
        handle = tf.placeholder(tf.string, name="iterator handle", shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle,
                                                       validation_dataset.output_types, 
                                                       validation_dataset.output_shapes)
        
        images, labels = iterator.get_next()

        validation_iterator = validation_dataset.make_initializable_iterator()
        training_iterator = training_dataset.make_initializable_iterator()
        
        logits = rotnet.rotnet(images)
        loss = rotnet.loss(logits, labels)
        
        train_op = rotnet.train_op(loss, global_step)
        
        #Get the accuracy node in the graph to use for computing validation accuracy
        accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
        #Get the flag for whether the model is currently used for training or inference. 
        #(This is needed to perform batch normalization correctly)
        is_training = tf.get_default_graph().get_tensor_by_name("training_flag:0")
        #Create a file writer to log progress
        summary_writer = tf.summary.FileWriter("tmp/tmp")
        #Creates a saver that can save the model state.
        saver = tf.train.Saver()
        
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
                
                sess.run(train_op, feed_dict={handle : train_handle, is_training : True})

                if (i % LOG_INTERVAL == 0 or i == FLAGS.training_steps - 1):
                    summary = sess.run(tf.summary.merge_all(), feed_dict={handle : train_handle, is_training : True})
                    summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                    
                    val_loss = 0
                    val_acc = 0
                    #Compute the training and validation losses and accuracies by iterating over the validation set
                    #in batches and summing the result
                    #Log the values to view in Tensorboard later.
                    for _ in range(num_validation_iters):
                        val_loss += sess.run(loss, feed_dict={handle : val_handle, is_training : False}) / num_validation_iters
                        val_acc += sess.run(accuracy, feed_dict={handle : val_handle, is_training : False}) / num_validation_iters

                    #Log the values for viewing in tensorboard later.
                    _log_scalar(val_loss, "validation loss", i, summary_writer)
                    _log_scalar(val_acc, "validation accuracy", i, summary_writer)
                
                #Save the model variables to use for evaluation / training another model.
                if (i == FLAGS.training_steps - 1):
                    saver.save(sess, FLAGS.checkpoint_path + "/checkpoint.ckpt")
if __name__ == "__main__":          
    #train()
    eval(FLAGS.checkpoint_path + "/checkpoint.ckpt")