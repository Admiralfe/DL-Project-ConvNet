import math
import random
import os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import rotnet
import data

#Project wide constants and flags

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("training_steps", 30000,
                            """Number of steps to run training for""")
tf.app.flags.DEFINE_string("checkpoint_path", "tmp/checkpoints",
                           """Directory to save model variables to after training""")

#Global training constants
NUM_EPOCHS = 1
VALIDATION_BATCH_SIZE = 100
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
        
        logits = rotnet.rotnet(images, num_blocks=4)
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
        
        handle = tf.placeholder(tf.string, name="iterator_handle", shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle,
                                                       validation_dataset.output_types, 
                                                       validation_dataset.output_shapes)
        
        images, labels = iterator.get_next(name="iterator_outputs")

        validation_iterator = validation_dataset.make_initializable_iterator()
        training_iterator = training_dataset.make_initializable_iterator()
        
        #Build the graph
        logits = rotnet.rotnet(images, num_blocks=4)
        loss = rotnet.loss(logits, labels)
        
        train_op = rotnet.train_op(loss, global_step)
        
        #Get the accuracy node in the graph to use for computing validation accuracy
        accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
        #Get the flag for whether the model is currently used for training or inference. 
        #(This is needed to perform batch normalization correctly)
        is_training = tf.get_default_graph().get_tensor_by_name("training_flag:0")
        #Create a file writer to log progress
        if (os.path.isdir("tmp/features")):
            raise RuntimeError("The log file to write to already exists")
        summary_writer = tf.summary.FileWriter("tmp/features")
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
                    saver.save(sess, FLAGS.checkpoint_path + "/feature_model")
                    
def train_from_features(checkpoint_dir, num_samples_to_keep):
    training_steps = 6000
    
    if (os.path.isdir("tmp/classifier")):
        raise RuntimeError("The log file to write to already exists")
    summary_writer = tf.summary.FileWriter("tmp/classifier")
    
    print("load old graph...")
    logits = rotnet.make_final_classifier(checkpoint_dir)
    print("loading data...")
    training_images, training_labels = data.load_cifar_training_data()
    validation_images, validation_labels = data.load_cifar_validation_data()
    test_images, test_labels = data.load_cifar_test_data()
    training_images, training_labels = data.keep_k(training_images, training_labels, 10, num_samples_to_keep)
    validation_images, validation_labels = data.keep_k(validation_images, validation_labels, 10, num_samples_to_keep)
    print("creating pipelines...")
    training_pipeline = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    validation_pipeline = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    test_pipeline = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    #Apply random crops and left right flips to training data to improve
    #generalization
    training_pipeline = data.pre_process_data(training_pipeline)
    training_pipeline = training_pipeline.prefetch(1).shuffle(100).repeat().batch(FLAGS.batch_size)
    
    validation_pipeline = validation_pipeline.map(data.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_pipeline = validation_pipeline.prefetch(1).repeat().batch(VALIDATION_BATCH_SIZE)
    
    test_pipeline = test_pipeline.map(data.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_pipeline = test_pipeline.prefetch(1).batch(VALIDATION_BATCH_SIZE)
    
    training_iterator = training_pipeline.make_one_shot_iterator()
    validation_iterator = validation_pipeline.make_one_shot_iterator()
    test_iterator = test_pipeline.make_one_shot_iterator()
    #The number of iterations it takes to go through the validation/test iterator once.
    num_validation_iters = math.ceil(len(validation_labels) / VALIDATION_BATCH_SIZE)
    num_test_iters = math.ceil(len(test_labels) / VALIDATION_BATCH_SIZE)

    #The old saved graph got its input iterator by string handle
    #Feeding that graph the string handle of this iterator will make it use this iterator
    #as its new input pipeline.
    pretrained_graph_input_handle = tf.get_default_graph().get_tensor_by_name("iterator_handle:0")
    get_next_op = tf.get_default_graph().get_operation_by_name("iterator_outputs")
    images, labels = get_next_op.outputs
    
    is_training = tf.get_default_graph().get_tensor_by_name("training_flag:0")
    
    with tf.variable_scope("Cifar_classification"):
        global_step = tf.train.get_or_create_global_step()
        step_reset_op = global_step.assign(0)
        #Compute the new loss function
        labels = tf.cast(labels, tf.int32)
        sample_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                               labels=labels, 
                                               logits=logits, 
                                               name="cross_entropy_per_sample")
        cross_entropy = tf.reduce_mean(sample_cross_entropy, name="cross_entropy")
        
        tf.add_to_collection("cifar_10_losses", cross_entropy)
        correct_predictions = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        tf.summary.scalar("Training accuracy", accuracy)
        
        loss = tf.add_n(tf.get_collection("cifar_10_losses"), name="total_loss")
        
        train_op = rotnet.train_op(loss, global_step)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(step_reset_op)
        train_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        
        summary_writer.add_graph(sess.graph)
        
        for i in range(training_steps):
            print("starting iteration " + str(i) + " of " + str(training_steps))
            sess.run(train_op, feed_dict={pretrained_graph_input_handle : train_handle, is_training : True})
            if (i % LOG_INTERVAL == 0 or i == FLAGS.training_steps - 1):
                summary = sess.run(tf.summary.merge_all(), feed_dict={pretrained_graph_input_handle : train_handle, is_training : True})
                summary_writer.add_summary(summary, i)
                val_loss = 0
                val_acc = 0
                #Compute the training and validation losses and accuracies by iterating over the validation set
                #in batches and summing the result
                #Log the values to view in Tensorboard later.
                for _ in range(num_validation_iters):
                    val_loss += sess.run(loss, feed_dict={pretrained_graph_input_handle : validation_handle, is_training : False}) / num_validation_iters
                    val_acc += sess.run(accuracy, feed_dict={pretrained_graph_input_handle : validation_handle, is_training : False}) / num_validation_iters

                #Log the values for viewing in tensorboard later.
                _log_scalar(val_loss, "validation loss", i, summary_writer)
                _log_scalar(val_acc, "validation accuracy", i, summary_writer)
                        #Save the model variables to use for evaluation / training another model.
        
        test_acc = 0
        #Training finishes here so we can run the evaluation
        for i in range(num_test_iters):
            test_acc += sess.run(accuracy, feed_dict={pretrained_graph_input_handle : test_handle, is_training : False}) / num_test_iters
    print("Final test accuracy: ", test_acc)
    _log_scalar(test_acc, "Test accuracy", training_steps - 1, summary_writer)
    return        
if __name__ == "__main__":          
    #train()
    eval(FLAGS.checkpoint_path + "/feature_model")
    #train_from_features(FLAGS.checkpoint_path, 100)