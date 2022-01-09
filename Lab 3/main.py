from PIL import Image

import cv2
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tabulate import tabulate

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# BUILDING THE CONVOLUTIONAL NEURAL NETWORK FOLLOWING THE TUTORIAL

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Plot the first 25 images from the training set and display the class name below each image
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

print("Building the CNN following the tutorial")

# Define the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) # converting the output from previous layer from 3D to 1D
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

def train(model,train_images,train_labels,test_images,test_labels):
    history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
    return history

history=train(model,train_images,train_labels,test_images,test_labels)

# Evaluate the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc_tutorial = model.evaluate(test_images,  test_labels, verbose=2)

print("Test accuracy:"+str(test_acc_tutorial))

model.save('first_tensorflow_model')

########################################################################
# Change the initializers of the layers with ReLu activations to He initializer

print("Change the initializers of the layers with ReLu activations to He initializer")

initializer = tf.keras.initializers.HeNormal()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                        kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_initializer=initializer))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu',
                       kernel_initializer=initializer))
model.add(layers.Dense(10))

#model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc_he = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy:"+str(test_acc_he))

model.save('he_tensorflow_model')

########################################################################
# Add some regularization to your network

print("Add some regularization to your network")

from tensorflow.keras import regularizers

initializer = tf.keras.initializers.HeNormal()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5),
                        kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5),
                        kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5),
                        kernel_initializer=initializer))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                       bias_regularizer=regularizers.l2(1e-4),
                       activity_regularizer=regularizers.l2(1e-5),
                       kernel_initializer=initializer))
model.add(layers.Dense(10))

#model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc_regularization = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy with regularization"+str(test_acc_regularization))

model.save('he_tensorflow_model_regularization')

########################################################################
# Add a dropout layer after the layer with the highest number of parameters and retrain your network

print("Add a dropout layer after the layer with the highest number of parameters and retrain your network")

from tensorflow.keras import regularizers

initializer = tf.keras.initializers.HeNormal()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5),
                        kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5),
                        kernel_initializer=initializer))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5),
                        kernel_initializer=initializer))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                       bias_regularizer=regularizers.l2(1e-4),
                       activity_regularizer=regularizers.l2(1e-5),
                       kernel_initializer=initializer))
model.add(layers.Dropout(.3))
model.add(layers.Dense(10))

#model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc_dropout = model.evaluate(test_images, test_labels, verbose=2)

print("Test accuracy with regularization and dropout:"+str(test_acc_dropout))

model.save('he_tensorflow_model_regularization_and_dropout')

########################################################################
# Creating the table of test accuracy
table=[["Version","Test accuracy"],["Tutorial",test_acc_tutorial],["He init", test_acc_he],
       ["Regularization", test_acc_regularization],["Reg+Dropout",test_acc_dropout]]
print(tabulate(table))

########################################################################
# Write a custom layer called Cutout

print("Write a custom layer called Cutout")
import numpy as np


def cutout(x, cutout_size):
    shape = x.get_shape()
    mask = np.ones(shape)
    width, height = shape[0], shape[1]

    x_coord = np.random.randint(width)
    y_coord = np.random.randint(height)

    tl_x = np.clip(x_coord - cutout_size // 2, 0, width)
    tl_y = np.clip(y_coord - cutout_size // 2, 0, height)
    br_x = np.clip(x_coord + cutout_size // 2, 0, width)
    br_y = np.clip(y_coord + cutout_size // 2, 0, height)

    mask[tl_x:br_x, tl_y:br_y, :] = np.zeros((br_x - tl_x, br_y - tl_y, shape[2]))
    # print(mask)
    return tf.where(tf.convert_to_tensor(mask, dtype=tf.bool), x, 0)


class Cutout(layers.Layer):
    def __init__(self, cropSize, **kwargs):
        super().__init__(**kwargs)
        self.cropSize = cropSize  # cropped region will be cropSize*2+1

    def call(self, x, training=None):
        if not training:
            return x
        return tf.map_fn(lambda elem: cutout(elem, self.cropSize), x)


class CutoutModel(tf.keras.Model):
    def __init__(self, cropSize, input_shape):
        super(CutoutModel, self).__init__(name='')
        self.cutout = Cutout(cropSize, input_shape=input_shape)

    def call(self, input_tensor, training=False):
        x = input_tensor
        if training:
            x = self.cutout(input_tensor)
        return x

# test cutout
image = cv2.imread('cameraman.jpg')
image = cv2.resize(image, (500, 500))
# X contains a single image sample
X = np.stack([image]*32)
print(X.shape)
cut_X = Cutout(100)(tf.convert_to_tensor(X))
# print(cut_X)

plt.subplot(1,3,1)
plt.imshow(image)
plt.subplot(1,3,2)
plt.imshow(cut_X[0])
plt.subplot(1,3,3)
plt.imshow(cut_X[1])














def zero_pad(X, pad):
    """
    This function applies the zero padding operation on all the images in the array X
    :param X input array of images; this array has a of rank 4 (batch_size, height, width, channels)
    :param pad the amount of zeros to be added around around the spatial size of the images
    """
    # hint you might find the function numpy.pad useful for this purpose
    # keep in mind that you only need to pad the spatial dimensions (height and width)
    # TODO your code here
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))


# load the image using Pillow
img = Image.open('cameraman.jpg')
img = np.asarray(img)

# TODO your code here
# pad and display the cameraman.jpg image
# (if you are using matplotlib to display the image, use cmap='gray' in the imshow function)
print(img.shape)
img = np.stack([np.stack([img], axis=0)], axis=-1)
print(img.shape)
img = zero_pad(img, 50)[0, :, :, 0]
print(img.shape)
plt.imshow(img, cmap='gray')


# -------------------------------------------------------------------------------


def convolution(X, W, bias, pad, stride):
    """
    This function applied to convolution operation on the input X of shape (num_samples, iH, iW, iC)
    using the filters defined by the W (filter weights) and  (bias) parameters.

    :param X - input of shape (num_samples, iH, iW, iC)
    :param W - weights, numpy array of shape (fs, fs, iC, k), where fs is the filter size,
      iC is the depth of the input volume and k is the number of filters applied on the image
    :param bias - numpy array of shape (1, 1, 1, iC)
    :param pad - hyperparameter, the amount of padding to be applied
    :param stride - hyperparameter, the stride of the convolution
    """

    # 0. compute the size of the output activation map and initialize it with zeros

    num_samples = X.shape[0]
    iW = X.shape[2]
    iH = X.shape[1]
    f = W.shape[0]

    # TODO your code here
    # compute the output width (oW), height (oH) and number of channels (oC)
    oW = (iW - f + 2 * pad) // stride + 1
    oH = (iH - f + 2 * pad) // stride + 1
    oC = W.shape[3]
    # initialize the output activation map with zeros
    activation_map = np.zeros((num_samples, oH, oW, oC))
    # end TODO your code here

    # 1. pad the samples in the input
    # TODO your code here, pad X using pad amount
    X_padded = zero_pad(X, pad)
    # end TODO your code here

    # go through each input sample
    for i in range(num_samples):
        # TODO: get the current sample from the input (use X_padded)
        X_i = X_padded[i]
        # end TODO your code here

        # loop over the spatial dimensions
        for y in range(oH):
            # TODO your code here
            # compute the current ROI in the image on which the filter will be applied (y dimension)
            # tl_y - the y coordinate of the top left corner of the current region
            # br_y - the y coordinate of the bottom right corner of the current region
            tl_y = y * stride
            br_y = tl_y + f
            # end TODO your code here

            for x in range(oW):
                # TODO your code here
                # compute the current ROI in the image on which the filter will be applied (x dimension)
                # tl_x - the x coordinate of the top left corner of the current region
                # br_x - the x coordinate of the bottom right corner of the current region
                tl_x = x * stride
                br_x = tl_x + f
                # end TODO your code here

                for c in range(oC):
                    # select the current ROI on which the filter will be applied
                    roi = X_padded[i, tl_y: br_y, tl_x: br_x, :]
                    w = W[:, :, :, c]
                    b = bias[:, :, :, c]

                    # TODO your code here
                    # apply the filter with the weights w and bias b on the current image roi

                    # A. compute the elementwise product between roi and the weights of the filters (np.multiply)
                    a = np.multiply(roi, w)
                    # B. sum across all the elements of a
                    a = np.sum(a)
                    # C. add the bias term
                    a = np.add(a, b)

                    # D. add the result in the appropriate position of the output activation map
                    activation_map[i, y, x, c] = a
                    # end TODO your code here
                assert (activation_map.shape == (num_samples, oH, oW, oC))
    return activation_map


np.random.seed(10)
# 100 samples of shape (iH=13, iW=21, iC=4)
X = np.random.randn(100, 13, 21, 4)

# 8 filters (last dimension) of shape (3, 3)
W = np.random.randn(3, 3, 4, 8)
b = np.random.randn(1, 1, 1, 8)

am = convolution(X, W, b, pad=1, stride=2)

print("----------------------")

print("am's mean =\n", np.mean(am))
print("am[1, 2, 3] =\n", am[3, 2, 1])

# -------------------------------------------------------------------------------

# load the image using Pillow
image = Image.open('cameraman.jpg')
image = np.asarray(image)
image = np.expand_dims(image, axis=-1)

# X contains a single image sample
X = np.expand_dims(image, axis=0)

############################################################
# MEAN FILTER
############################################################

bias = np.asarray([0])
bias = bias.reshape((1, 1, 1, 1))

mean_filter_3 = np.ones(shape=(3, 3, 1, 1), dtype=np.float32)
mean_filter_3 = mean_filter_3 / 9.0

mean_filter_9 = np.ones(shape=(9, 9, 1, 1), dtype=np.float32)
mean_filter_9 = mean_filter_9 / 81.0

mean_3x3 = convolution(X, mean_filter_3, bias, pad=0, stride=1)
mean_9x9 = convolution(X, mean_filter_9, bias, pad=0, stride=1)

plt.figure(0)
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(mean_3x3[0, :, :, 0], cmap='gray')
plt.title('mean filter 3x3')

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(mean_9x9[0, :, :, 0], cmap='gray')
plt.title('mean filter 9x9')

############################################################
# GAUSSIAN FILTER
############################################################

gaussian_filter = np.asarray(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]],
    dtype=np.float32
)
gaussian_filter = gaussian_filter.reshape(3, 3, 1, 1)
gaussian_filter = gaussian_filter / 16.0

gaussian_smoothed = convolution(X, gaussian_filter, bias, pad=0, stride=1)

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
plt.imshow(gaussian_smoothed[0, :, :, 0], cmap='gray')
plt.title('Gaussian filtered')

plt.show()

# -------------------------------------------------------------------------------

sobel_horiz = np.asarray([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])

sobel_vert = sobel_horiz.T

sobel_horiz = np.reshape(sobel_horiz, (3, 3, 1, 1))
sobel_vert = np.reshape(sobel_vert, (3, 3, 1, 1))

sobel_x = convolution(X, sobel_horiz, bias, 0, 1)
sobel_y = convolution(X, sobel_vert, bias, 0, 1)

plt.subplot(1, 3, 1)
plt.imshow(image[:, :, 0], cmap='gray')
plt.title('Original image')
plt.subplot(1, 3, 2)
plt.imshow(np.abs(sobel_x[0, :, :, 0]) / np.abs(np.max(sobel_x[0, :, :, 0])) * 255, cmap='gray')
plt.title('Sobel X')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(sobel_y[0, :, :, 0]) / np.abs(np.max(sobel_y[0, :, :, 0])) * 255, cmap='gray')
plt.title('Sobel Y')
plt.tight_layout()

plt.show()


# -------------------------------------------------------------------------------

def pooling(X, filter_size, stride, type):
    """
    Implements the pooling operation

    :param X - input volume of shape (num_samples, H, W, C)
    :param filter_size - the size of the pooling
    :param stride - the stride of the pooling operation
    :param type - can be 'max' or 'avg'; the type of the pooling operation to apply

    Returns the output of the pooling operation.
    """

    # TODO your code here implement the pooling operation
    # you can ispire yourself from the convolution implementation on how to organize your code

    # 0. compute the size of the output activation map and initialize it with zeros

    num_samples = X.shape[0]
    iW = X.shape[2]
    iH = X.shape[1]

    # TODO your code here
    # compute the output width (oW), height (oH) and number of channels (oC)
    oW = (iW - filter_size) // stride + 1
    oH = (iH - filter_size) // stride + 1
    oC = W.shape[3]
    # initialize the output activation map with zeros
    activation_map = np.zeros((num_samples, oH, oW, oC))
    # end TODO your code here

    # go through each input sample
    for i in range(num_samples):
        # loop over the spatial dimensions
        for y in range(oH):
            # TODO your code here
            # compute the current ROI in the image on which the filter will be applied (y dimension)
            # tl_y - the y coordinate of the top left corner of the current region
            # br_y - the y coordinate of the bottom right corner of the current region
            tl_y = y * stride
            br_y = tl_y + filter_size
            # end TODO your code here

            for x in range(oW):
                # TODO your code here
                # compute the current ROI in the image on which the filter will be applied (x dimension)
                # tl_x - the x coordinate of the top left corner of the current region
                # br_x - the x coordinate of the bottom right corner of the current region
                tl_x = x * stride
                br_x = tl_x + filter_size
                # end TODO your code here

                for c in range(oC):
                    # select the current ROI on which the filter will be applied
                    roi = X[i, tl_y: br_y, tl_x: br_x, :]

                    if type == "max":
                        activation_map[i, y, x, c] = np.max(roi)
                    if type == "average":
                        activation_map[i, y, x, c] = np.average(roi)

                assert (activation_map.shape == (num_samples, oH, oW, oC))
    return activation_map


# TODO your code here
# apply the pooling operation on a grayscale image and on a color image
# try different values for the stride and filter size. What do you observe?

img = Image.open("cameraman.jpg")
img = np.asarray(img)
originalImg = img

img = np.stack([np.stack([img], axis=0)], axis=-1)

# avg_img=pooling(img,5,2,'average')[0]
# max_img=pooling(img,5,2,'max')[0]

avg_img = pooling(img, 10, 2, 'average')[0]
max_img = pooling(img, 10, 2, 'max')[0]

plt.subplot(1, 3, 1)
plt.imshow(originalImg[:, :], cmap='gray')
plt.title("Original image")
plt.subplot(1, 3, 2)
plt.imshow(avg_img[:, :, 0], cmap='gray')
plt.title("Average Pooling")
plt.subplot(1, 3, 3)
plt.imshow(max_img[:, :, 0], cmap='gray')
plt.title("Max pooling")

plt.show()

# -------------------------------------------------------------------------------
