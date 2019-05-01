import numpy as np
from matplotlib import pyplot as plt

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
	
train_batch = np.load("Datasets/cifar-10-batches-py/data_batch_1", encoding="latin1")
image_data = train_batch["data"]
test_image = np.reshape(image_data[100], (3, 32, 32)).transpose([1, 2, 0])

test_image = rotate(test_image, 270)
plt.imshow(test_image, interpolation="nearest")
plt.show()



