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

def keep_k(images, labels, k):
    """ Get the first k images for each label in the same order as they are provided.
    
        Args:
            images - A numpy array of images
            labels - A numpy array of labels 0-3
            k      - An int > 0 of how many images to return per label
        Returns:
            images_k - The chosen images. len(images_k) <= k
            labels_k - The corresponding labels. len(labels_k) == len(images_k)
    """
    assert k >= 0

    chosen_indexes = labels == -1
    for label in range(4):
        condition = labels == label
        chosen_indexes |= condition & (np.cumsum(condition) <= k)

    images_k = np.compress(chosen_indexes, images)
    labels_k = np.compress(chosen_indexes, labels)

    assert len(images_k) == k * 4
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
    print(bs)
    
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
    """ Applies some pre processing to the dataset
        Namely, normalizes the images and applies randomly crops and left right flips
        to the images.
        
        Args:
            dataset - dataset to pre-process
        
        Returns:
            The processed dataset
    """
    #Helpers to make map() act only on the images and not the labels in the dataset
    def _random_left_right(x, y):
        return tf.image.random_flip_left_right(x), y
    def _random_crop(x, y):
        return tf.random_crop(x, size=[FLAGS.image_size, FLAGS.image_size, 3]), y

    #Normalize the images to have zero mean and unit stddev. 
    #Leaves the labels as they are.
    dataset = dataset.map(data.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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

"""Code below this point is not currently used but is kept for legacy reasons."""
    
"""
Rotates an image 0, 90, 180 or 270 degrees counter-clockwise
Image needs to be given as an N x N x 3 RGB-matrix.
"""
def rotate(image, degrees):
    if degrees == 0:
        return image
    if degrees == 90:
        #Transposing the image and flipping vertically same as rotating 90 degrees.
        image = np.transpose(image, (1, 0, 2))
        image = np.flipud(image)
        return image
    if degrees == 180:
        #Flipping vertically and then horizontally same as rotating 180 degrees.
        image = np.flipud(image)
        image = np.fliplr(image)
        return image
    if degrees == 270:
        #Flipping vertically and transposing same as 270 degrees
        image = np.flipud(image)
        image = np.transpose(image, (1, 0, 2))
        return image
    
    print("Amount of rotation needs to be 0, 90, 180 or 270 degrees")
    return

""" Loads the default CIFAR-10 dataset and creates 0, 90, 180 and 270 degree rotations of every image ordered like
        image_1_0, image_1_90, image_1_180, image_1_270, image_2_0, image_2_90 and so on.
        It then saves the new image array to files in float16 if FLAGS.use_fp16 is true and float32 otherwise. The label file is ordered in the same way.
"""
def create_rotated_data():
    _IMAGE_SIZE = 32
    _NUM_CHANNELS = 3
    dtype = np.float16 if FLAGS.use_fp16 else np.float32
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_cifar_data()

    # Create all rotations of the training data
    training_shape = (train_data.shape[0] * 4, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)

    training_data_rotated = np.empty(training_shape, dtype)
    training_labels_rotated = np.empty(training_shape[0], np.int8)

    for i in range(train_data.shape[0]):
        for j in range(4):
            reshaped_img = np.reshape(train_data[i], (3, 32, 32)).transpose([1, 2, 0])# / 255.0
            training_data_rotated[i * 4 + j] = rotate(reshaped_img, j * 90)
            training_labels_rotated[i * 4 + j] = np.int8(j)

    np.save("training_data_rotated", training_data_rotated)
    np.save("training_labels_rotated", training_labels_rotated)

    # Create all rotations of the validation data
    val_shape = (val_data.shape[0] * 4, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)

    val_data_rotated = np.empty(val_shape, dtype)
    val_labels_rotated = np.empty(val_shape[0], np.int8)

    for i in range(val_data.shape[0]):
        for j in range(4):
            reshaped_img = np.reshape(val_data[i], (3, 32, 32)).transpose([1, 2, 0])# / 255.0
            val_data_rotated[i * 4 + j] = rotate(reshaped_img, j * 90)
            val_labels_rotated[i * 4 + j] = np.int8(j)

    np.save("val_data_rotated", val_data_rotated)
    np.save("val_labels_rotated", val_labels_rotated)

    # Create all rotations of the test data
    test_shape = (test_data.shape[0] * 4, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)

    test_data_rotated = np.empty(test_shape, dtype)
    test_labels_rotated = np.empty(test_shape[0], np.int8)

    for i in range(test_data.shape[0]):
        for j in range(4):
            reshaped_img = np.reshape(test_data[i], (3, 32, 32)).transpose([1, 2, 0])# / 255.0
            test_data_rotated[i * 4 + j] = rotate(reshaped_img, j * 90)
            test_labels_rotated[i * 4 + j] = np.int8(j)

    np.save("test_data_rotated", test_data_rotated)
    np.save("test_labels_rotated", test_labels_rotated)
    
def load_training_data():
    train_data = np.load("training_data_rotated.npy", encoding="latin1", allow_pickle=True)
    train_labels = np.load("training_labels_rotated.npy", encoding="latin1", allow_pickle=True)
    
    tf.app.flags.DEFINE_integer("num_training_samples", len(train_labels),
                                """Number of samples in the training set""")

    return train_data, train_labels

def load_validation_data():
    val_data = np.load("val_data_rotated.npy", encoding="latin1", allow_pickle=True)
    val_labels = np.load("val_labels_rotated.npy", encoding="latin1", allow_pickle=True)
    
    return val_data, val_labels

def load_test_data():
    test_data = np.load("test_data_rotated.npy", encoding="latin1", allow_pickle=True)
    test_labels = np.load("test_labels_rotated.npy", encoding="latin1", allow_pickle=True)
    
    return test_data, test_labels
    
"""
    Test code for this file
"""
if __name__ == "__main__":
    create_rotated_data()
    #from random import randint
    #imgs = np.array(range(100))
    #labels = np.array([randint(0, 3) for x in range(len(imgs))])
    #print(imgs)
    #print(labels)
    #img_k, labels_k = keep_k(imgs, labels, 3)
    #print(img_k)
    #print(labels_k)

"""

train_batch = np.load("Datasets/cifar-10-batches-py/data_batch_1", encoding="latin1")
image_data = train_batch["data"]
test_image = np.reshape(image_data[100], (3, 32, 32)).transpose([1, 2, 0])

test_image = rotate(test_image, 270)
plt.imshow(test_image, interpolation="nearest")
plt.show()
"""
