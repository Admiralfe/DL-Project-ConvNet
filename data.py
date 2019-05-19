import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("image_size", 32,
                            """Size of the square images in the data set""")
#Global constants
IMAGE_SIZE = 32;
IMAGE_CHANNELS = 3;

""" Loads the CIFAR-10-data, where the first 4 data batches are used as training data,
    and the fifth is used as validation.
"""
def load_cifar_data():
    """ Loads all the cifar-10 data, using 45000 samples for training and 5000 for validation
        
        Returns:
            Training, validation and test labels and images.
    """
    data_path_root = "Datasets/cifar-10-batches-py/"
    train_batch_1 = np.load(data_path_root + "data_batch_1", encoding="latin1", allow_pickle=True)
    train_batch_2 = np.load(data_path_root + "data_batch_2", encoding="latin1", allow_pickle=True)
    train_batch_3 = np.load(data_path_root + "data_batch_3", encoding="latin1", allow_pickle=True)
    train_batch_4 = np.load(data_path_root + "data_batch_4", encoding="latin1", allow_pickle=True)
    train_batch_5 = np.load(data_path_root + "data_batch_5", encoding="latin1", allow_pickle=True)
    test_batch = np.load(data_path_root + "test_batch", encoding="latin1", allow_pickle=True)
    
    #Use the first 45000 samples for training and leave 5000 for validation
    train_data = np.concatenate((train_batch_1["data"], train_batch_2["data"], train_batch_3["data"], train_batch_4["data"], train_batch_5["data"][0:5000]))
    train_labels = np.concatenate((train_batch_1["labels"], train_batch_2["labels"], train_batch_3["labels"], train_batch_4["labels"], train_batch_5["labels"][0:5000]))
    val_data = train_batch_5["data"][5000:]
    val_labels = train_batch_5["labels"][5000:]
    
    test_data = test_batch["data"]
    test_labels = test_batch["labels"]
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def load_cifar_training_data():
    """ Loads only the training data of the cifar-10 dataset consisting of 45000 images.
        The last 5000 images are left to be used as a validation set.
        
        Returns:
            Numpy arrays with training images in 32x32x3 format and training labels.
    """
    dtype = np.float16 if FLAGS.use_fp16 else np.float32
    
    data_path_root = "Datasets/cifar-10-batches-py/"
    train_batch_1 = np.load(data_path_root + "data_batch_1", encoding="latin1", allow_pickle=True)
    train_batch_2 = np.load(data_path_root + "data_batch_2", encoding="latin1", allow_pickle=True)
    train_batch_3 = np.load(data_path_root + "data_batch_3", encoding="latin1", allow_pickle=True)
    train_batch_4 = np.load(data_path_root + "data_batch_4", encoding="latin1", allow_pickle=True)
    train_batch_5 = np.load(data_path_root + "data_batch_5", encoding="latin1", allow_pickle=True)
    
    train_data = np.concatenate((train_batch_1["data"], train_batch_2["data"], train_batch_3["data"], train_batch_4["data"], train_batch_5["data"][0:5000]))
    train_labels = np.concatenate((train_batch_1["labels"], train_batch_2["labels"], train_batch_3["labels"], train_batch_4["labels"], train_batch_5["labels"][0:5000]))

    array_shape = [train_data.shape[0], 32, 32, 3]
    reshaped_train_data = np.empty(array_shape, dtype)
    
    for i in range(train_data.shape[0]):
        reshaped_train_data[i] = np.reshape(train_data[i], (3, 32, 32)).transpose([1, 2, 0])
        
    return reshaped_train_data, train_labels
    
def load_cifar_validation_data():
    """ Loads the validation data of the cifar-10 dataset consisting of 5000 images.
        
        Returns:
            Numpy arrays with validation images in 32x32x3 format and validation labels.
    """
    dtype = np.float16 if FLAGS.use_fp16 else np.float32
    
    data_path_root = "Datasets/cifar-10-batches-py/"
    val_batch = np.load(data_path_root + "data_batch_5", encoding="latin1", allow_pickle=True)
    val_data = val_batch["data"][5000:]
    val_labels = np.asarray(val_batch["labels"][5000:])
    
    array_shape = [val_data.shape[0], 32, 32, 3]
    reshaped_val_data = np.empty(array_shape, dtype)
    
    for i in range(val_data.shape[0]):
        reshaped_val_data[i] = np.reshape(val_data[i], (3, 32, 32)).transpose([1, 2, 0])
    return reshaped_val_data, val_labels
    
def load_cifar_test_data():
    """ Loads the test data of the cifar-10 dataset consisting of 10000 images.
        
        Returns:
            Numpy arrays with test images in 32x32x3 format and test labels.
    """
    dtype = np.float16 if FLAGS.use_fp16 else np.float32
    
    data_path_root = "Datasets/cifar-10-batches-py/"
    test_batch = np.load(data_path_root + "test_batch", encoding="latin1", allow_pickle=True)
    test_data = test_batch["data"]
    test_labels = np.asarray(test_batch["labels"])
    
    array_shape = [test_data.shape[0], 32, 32, 3]
    reshaped_test_data = np.empty(array_shape, dtype)
    
    for i in range(test_data.shape[0]):
        reshaped_test_data[i] = np.reshape(test_data[i], (3, 32, 32)).transpose([1, 2, 0])
    return reshaped_test_data, test_labels
    
def make_tf_dataset(image_shape, labels_shape):
    """ Builds a tf dataset using placeholders for the input data
        
        Adds the placeholders to a collection called iterator_inputs for access later
        
        Args:
            image_shape - shape of the input images
            labels_shape - shape of the input labels
        
        Returns:
            A tf.data.Dataset with placeholders for the dataset.
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    images_input = tf.placeholder(dtype=dtype, shape=image_shape)
    labels_input = tf.placeholder(dtype=tf.uint8, shape=labels_shape)
    
    tf.add_to_collection("iterator_inputs", images_input)
    tf.add_to_collection("iterator_inputs", labels_input)   
    
    return tf.data.Dataset.from_tensor_slices((images_input, labels_input))
    
def create_rotated_images_with_labels(image, curr_labels):
    """ Creates 4 copies of the input image, rotated by 0, 90, 180 and 270 degrees.
        
        Args:
            image - image to rotate
            curr_labels - the current labels in the dataset 
            (this argument is for compatibility with the tf.data.Dataset.map() function)
        Returns:
            the four rotated images
    """
    rotated = [image, 
               tf.image.rot90(image, k=1), 
               tf.image.rot90(image, k=2), 
               tf.image.rot90(image, k=3)]
    return tf.stack(rotated), tf.range(4, dtype=tf.int32)

def keep_k(images, labels, num_labels, k):
    """ Get the first k images for each label in the same order as they are provided.
    
        Args:
            images - A numpy array of images
            labels - A numpy array of labels in range 0 - num_labels
            num_labels - number of labels
            k      - An int > 0 of how many images to return per label
        Returns:
            images_k - The chosen images. len(images_k) <= k
            labels_k - The corresponding labels. len(labels_k) == len(images_k)
    """
    assert k >= 0

    chosen_indexes = labels == -1
    for label in range(num_labels):
        condition = labels == label
        chosen_indexes |= condition & (np.cumsum(condition) <= k)
        
    images_k = np.compress(chosen_indexes, images, axis=0)
    labels_k = np.compress(chosen_indexes, labels, axis=0)
    
    assert len(images_k) == k * num_labels
    return images_k, labels_k

def data_pipeline(dataset, batch_size=None):
    """ Makes a data pipeline for a certain number of epochs for training the model.
        !!!The iterator returned needs to be initialized manually with the input data!!!
        
        Args:
            dataset - dataset to create iterator over
            num_epochs (optional) - number of epochs to create iterator for,
                                    if not specified then use FLAGS.num_epochs
            batch_size (optional) - size of batches in iterator,
                                    if not specified then use FLAGS.batch_size
        Returns:
            Uninitialized iterator
            
    """
    bs = FLAGS.batch_size if batch_size is None else batch_size
    
    #Shuffles the data in blocks of 4 so that 
    #rotated versions of the same image stay next to each other in the data set
    print("Shuffling and repeating dataset...")
    dataset = dataset.batch(4).shuffle(buffer_size=1000).repeat()
    
    #Undo the batching used to keep rotated images together
    dataset = dataset.apply(tf.data.experimental.unbatch())
    
    #Re-batch the data into the training batch size.
    print("Re-batching dataset...")
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(1)
    
    iterator = dataset.make_initializable_iterator()
    return iterator

def pre_process_data(dataset):
    """ Applies some pre processing to the dataset.
        Namely, normalizes the images and randomly applies crops and left right flips
        to the images.
        
        Args:
            dataset - dataset to pre-process
        
        Returns:
            The processed dataset
    """
    
    print(dataset.output_shapes)
    #Helpers to make map() act only on the images and not the labels in the dataset
    def _random_left_right(x, y):
        return tf.image.random_flip_left_right(x), y
    def _random_crop(x, y):
        return tf.random_crop(x, size=[FLAGS.image_size, FLAGS.image_size, 3]), y
    
    #Normalize the images to have zero mean and unit stddev. 
    #Leaves the labels as they are.
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    padding = tf.constant([[4, 4], [4, 4], [0, 0]])
    
    #Zero pad with 4 on each border for cropping later.
    dataset = dataset.map(lambda x, y: (tf.pad(x, padding), y))
    dataset = dataset.map(_random_crop, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #Apply random crops and left right flips to the images.
    dataset = dataset.map(_random_left_right, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    return dataset

def normalize(img, label):
    """ Helper function to be used with tf.data.Dataset.map() for normalizing images
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    return tf.cast(tf.image.per_image_standardization(img), dtype), label

    
"""
    Test code for this file
"""
if __name__ == "__main__":
    create_rotated_data()
