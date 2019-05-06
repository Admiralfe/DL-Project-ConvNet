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
def load_data():
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

train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data()

"""
train_batch = np.load("Datasets/cifar-10-batches-py/data_batch_1", encoding="latin1")
image_data = train_batch["data"]
test_image = np.reshape(image_data[100], (3, 32, 32)).transpose([1, 2, 0])

test_image = rotate(test_image, 270)
plt.imshow(test_image, interpolation="nearest")
plt.show()
"""
