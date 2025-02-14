import tensorflow as tf
import numpy as np
import rotnet
import rotation
import os
from datetime import datetime


#Test if the forward pass of the graph works.
def test_graph():
    train_data, train_labels, val_data, val_labels, test_data, test_labels = rotation.load_data()
    test_image = np.reshape(train_data[0], (1, 3, 32, 32)).transpose([0, 2, 3, 1])
    test_label = np.zeros((1, 4))
    test_label[test_labels[0]] = 1
    x, y, probs, logits = rotnet.rotnet()
    with tf.Session() as sess:
        now =  datetime.strftime(datetime.now(), "%y%m%d_%H%M%S")
        file_writer = tf.summary.FileWriter('logs/test/' + now, sess.graph)
        sess.run(tf.global_variables_initializer())
        probs, logits = sess.run([probs, logits], feed_dict={x: test_image, y: test_label})
        print(probs)
    return

def test_training():
    train_data, train_labels, val_data, val_labels, test_data, test_labels = rotation.load_data()
    train_onehot = np.eye(4)[train_labels]
    logits = rotnet.rotnet()
    inputs_collection = tf.get_collection("inputs")
    x = inputs_collection[0]
    y = inputs_collection[1]
    loss = rotnet.loss(logits, train_labels)
    with tf.Session() as sess:
        now =  datetime.strftime(datetime.now(), "%y%m%d_%H%M%S")
        file_writer = tf.summary.FileWriter('logs/train/' + now, sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        curr_loss = sess.run([loss, merged], feed_dict={x : train_data[0:100], y : train_onehot[0:100]})
        file_writer.add_summary(merged)
        print(curr_loss)
    rotnet.train(train_data, train_labels, logits)
# Suppress tensorflow messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def make_graph_visualization():
    global_step = tf.train.get_or_create_global_step()

    training_images, training_labels = rotation.load_training_data()

    dataset = rotation.make_tf_dataset(images_shape=training_images.shape, labels_shape=training_labels.shape)
    images, labels, iterator_initializer = rotation.data_pipeline(dataset)
    
    logits = rotnet.rotnet(images)
    loss = rotnet.loss(logits, labels)
    
    train_op = rotnet.train_op(loss, global_step)
    
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("tmp/1")
        writer.add_graph(sess.graph)
    return

#test_training()