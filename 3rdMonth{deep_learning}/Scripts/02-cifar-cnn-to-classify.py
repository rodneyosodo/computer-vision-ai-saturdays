#!/usr/bin/env python
# coding: utf-8

# # Building a CNN to classify images on the CIFAR dataset

# https://www.cs.toronto.edu/~kriz/cifar.html|
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

# ## Building a CNN
# We will check how different layers are configured then build our own model

# In[11]:


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Deep learning framework
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
# For plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append("..")
from Scripts.utils import plot_loss_accuracy


# In[2]:


# Loading the data and splitting it to train and test
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[3]:


# Lets look at one of the images and its label
print(y_train[0])
plt.imshow(x_train[0])


# In[4]:


classes = 10

y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)


# In[5]:


print(y_train[0])


# In[6]:


# Making everything to be float and scale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# ### Conv2D
# 
# ```python
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# ```
# When using this layer as the first layer in a model, provide the keyword argument input_shape
# 
# - `filters`: Integer, the number of output filters in the convolution.
# - `kernel_size`: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
# - `strides`: and (x,y) tuple giving the stride in each dimension.  Default is `(1,1)`
# - `activation`: Activation function to use. If you don't specify anything, no activation is applied.
# 
# 
# Note, the size of the output will be determined by the kernel_size, strides
# 
# ### MaxPooling2D
# ```python
# keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# ```
# 
# - `pool_size`: Integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal).
# - `strides`: Integer, tuple of 2 integers, or None. Strides values.
# 
# ### Flatten
# ```python
# keras.layers.Flatten(data_format=None)
# ```
# Turns its input into a one-dimensional vector (per instance). Usually used when transitioning between convolutional layers and fully connected layers.
# 
# ### Activation
# ```python
# keras.layers.Activation(activation)
# ```
# 
# Applies an activation function to the output
# 
# ### Sequential
# The Sequential model is a linear stack of layers. You can create a Sequential model by passing a list of layer instances to the constructor
# 
# 

# In[7]:



model_1 = Sequential()


model_1.add(Conv2D(64, (4, 4), strides = (2,2), padding='same', input_shape=x_train.shape[1:]))
model_1.add(Activation('relu'))
model_1.add(Conv2D(64, (4, 4), strides = (2,2)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.3))
model_1.add(Flatten())
model_1.add(Dense(512))
model_1.add(Activation('relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(classes))
model_1.add(Activation('softmax'))

model_1.summary()


# In[8]:


opt = keras.optimizers.rmsprop(lr=0.0005, decay=1e-6)
model_1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model_1.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test), shuffle=True)


# In[9]:


score = model_1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[12]:


plot_loss_accuracy(model_1.history)


# ### Exercise
# Our previous model had the structure:
# 
# Conv -> Conv -> MaxPool -> Flatten -> Dense -> Final Classification
# 
# (with appropriate activation functions and dropouts)
# 
# 1. Build a more complicated model with the following pattern:
# - Conv -> Conv -> Conv -> MaxPool -> Conv -> Conv -> MaxPool -> Flatten -> Dense -> Dense -> Final Classification
# 
# - Use strides of 1 for all convolutional layers.
# 
# 2. How many parameters does your model have

# In[ ]:





# In[ ]:





# In[16]:


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images
                   
datagen.fit(x_train)      
# This computes any statistics that may be needed (e.g. for centering) from the training set.

    
# Fit the model on the batches generated by datagen.flow().
model_1.fit_generator(datagen.flow(x_train, y_train,batch_size=128), steps_per_epoch=x_train.shape[0] // 128, epochs=10, validation_data=(x_test, y_test))


# In[17]:


plot_loss_accuracy(model_1.history)


# In[ ]:




