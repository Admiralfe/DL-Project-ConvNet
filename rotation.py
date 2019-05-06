import numpy as np
from matplotlib import pyplot as plt

#TODO: create a function to load all images, rotate them and save on disk.
"""
Rotates an image 0, 90, 180 or 270 degrees counter-clockwise
Image needs to be given as a N x N x 3 RGB-matrix.
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

""" Loads the CIFAR-10-data, where the first 4 data batches are used as training data,
	and the fifth is used as validation.
"""
def load_cifar_data():
	data_path_root = "Datasets/cifar-10-batches-py/"
	train_batch_1 = np.load(data_path_root + "data_batch_1", encoding="latin1", allow_pickle=True)
	train_batch_2 = np.load(data_path_root + "data_batch_2", encoding="latin1", allow_pickle=True)
	train_batch_3 = np.load(data_path_root + "data_batch_3", encoding="latin1", allow_pickle=True)
	train_batch_4 = np.load(data_path_root + "data_batch_4", encoding="latin1", allow_pickle=True)
	val_batch = np.load(data_path_root + "data_batch_5", encoding="latin1", allow_pickle=True)
	test_batch = np.load(data_path_root + "test_batch", encoding="latin1", allow_pickle=True)
	
	train_data = np.concatenate((train_batch_1["data"], train_batch_2["data"], train_batch_3["data"], train_batch_4["data"]))
	train_labels = np.concatenate((train_batch_1["labels"], train_batch_2["labels"], train_batch_3["labels"], train_batch_4["labels"]))
	val_data = val_batch["data"]
	val_labels = val_batch["labels"]
	
	test_data = test_batch["data"]
	test_labels = test_batch["labels"]
	
	return train_data, train_labels, val_data, val_labels, test_data, test_labels

""" Loads the default CIFAR-10 dataset and creates 0, 90, 180 and 270 degree rotations of every image ordered like
		image_1_0, image_1_90, image_1_180, image_1_270, image_2_0, image_2_90 and so on.
		It then saves the new image array to files in np.half for saving space. The label file is ordered in the same way.
"""
def create_rotated_data():
	train_data, train_labels, val_data, val_labels, test_data, test_labels = load_cifar_data()

	# Create all rotations of the training data
	training_shape = (train_data.shape[0] * 4, train_data.shape[1])

	training_data_rotated = np.empty(training_shape, np.half)
	training_labels_rotated = np.empty(training_shape[0], np.int8)

	for i in range(train_data.shape[0]):
		for j in range(4):
			reshaped_img = np.reshape(train_data[i], (3, 32, 32)).transpose([1, 2, 0]) / 255
			training_data_rotated[i * 4 + j] = rotate(reshaped_img, j * 90).transpose([2, 0, 1]).reshape(3072)
			training_labels_rotated[i * 4 + j] = np.int8(j)

	np.save("training_data_rotated", training_data_rotated)
	np.save("training_labels_rotated", training_labels_rotated)

	# Create all rotations of the validation data
	val_shape = (val_data.shape[0] * 4, val_data.shape[1])

	val_data_rotated = np.empty(val_shape, np.half)
	val_labels_rotated = np.empty(val_shape[0], np.int8)

	for i in range(val_data.shape[0]):
		for j in range(4):
			reshaped_img = np.reshape(val_data[i], (3, 32, 32)).transpose([1, 2, 0]) / 255
			val_data_rotated[i * 4 + j] = rotate(reshaped_img, j * 90).transpose([2, 0, 1]).reshape(3072)
			val_labels_rotated[i * 4 + j] = np.int8(j)

	np.save("val_data_rotated", val_data_rotated)
	np.save("val_labels_rotated", val_labels_rotated)

	# Create all rotations of the test data
	test_shape = (test_data.shape[0] * 4, test_data.shape[1])

	test_data_rotated = np.empty(test_shape, np.half)
	test_labels_rotated = np.empty(test_shape[0], np.int8)

	for i in range(test_data.shape[0]):
		for j in range(4):
			reshaped_img = np.reshape(test_data[i], (3, 32, 32)).transpose([1, 2, 0]) / 255
			test_data_rotated[i * 4 + j] = rotate(reshaped_img, j * 90).transpose([2, 0, 1]).reshape(3072)
			test_labels_rotated[i * 4 + j] = np.int8(j)

	np.save("test_data_rotated", test_data_rotated)
	np.save("test_labels_rotated", test_labels_rotated)

def load_data():
	train_data = np.load("training_data_rotated.npy", encoding="latin1", allow_pickle=True)
	train_labels = np.load("training_labels_rotated.npy", encoding="latin1", allow_pickle=True)

	val_data = np.load("val_data_rotated.npy", encoding="latin1", allow_pickle=True)
	val_labels = np.load("val_labels_rotated.npy", encoding="latin1", allow_pickle=True)

	test_data = np.load("test_data_rotated.npy", encoding="latin1", allow_pickle=True)
	test_labels = np.load("test_labels_rotated.npy", encoding="latin1", allow_pickle=True)

	return train_data, train_labels, val_data, val_labels, test_data, test_labels


"""
	Test code for this file
"""
create_rotated_data()

"""

train_batch = np.load("Datasets/cifar-10-batches-py/data_batch_1", encoding="latin1")
image_data = train_batch["data"]
test_image = np.reshape(image_data[100], (3, 32, 32)).transpose([1, 2, 0])

test_image = rotate(test_image, 270)
plt.imshow(test_image, interpolation="nearest")
plt.show()
"""
