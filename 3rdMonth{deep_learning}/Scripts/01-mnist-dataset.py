#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/0x6f736f646f/computer-vision-ai-saturdays/blob/master/3rdMonth%7Bdeep_learning%7D/Notebook/01-mnist-dataset.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Handwritten Image Detection with Keras using MNIST data

# We will load the data then build a network and train it. Finally we will try different models

# In[ ]:


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Deep learning framework
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
# Numerical computation
import numpy as np
# For plotting images and graphs
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from Scripts.utils import plot_loss_accuracy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Load the data and split it to train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[ ]:


# Checking shape for one of the images
x_train[0].shape


# In[ ]:


# Looking at actual data in image form
plt.imshow(x_train[0], cmap='gray')


# In[ ]:


# The actual label for the above image
y_train[0]


# In[ ]:


# Lets check the shape for our images
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# In[ ]:


# To feed MNIST instances into a neural network, they need to be reshaped, from a 2 dimensional image representation to a single dimension sequence
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)


# In[ ]:


# Casting to floats
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255


# In[ ]:


# Converting target to binary class matrix
classes = 10
y_train = keras.utils.to_categorical(y_train, classes)
y_test = keras.utils.to_categorical(y_test, classes)


# In[ ]:


y_train[0]


# In[ ]:


# Lets start with a simple model with 1 hidden layers
model_1 = Sequential()
model_1.add(Dense(64, activation="relu", input_shape=(784, )))
model_1.add(Dropout(0.3))
model_1.add(Dense(64, activation='relu'))
model_1.add(Dropout(0.3))
model_1.add(Dense(10, activation='softmax'))
model_1.summary()


# In[ ]:


# Let's compile the model
learning_rate = .001
model_1.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate), metrics=['accuracy'])


# In[ ]:


# And now let's fit.
history = model_1.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))


# In[ ]:


## We will use Keras evaluate function to evaluate performance on the test set

score = model_1.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


plot_loss_accuracy(history)


# ## Exercise
# ### Your Turn: Build your own model
# Use the Keras "Sequential" functionality to build `model_2` with the following specifications:
# 
# 1. Three hidden layers.
# 2. Layers
#  - First 300
#  - Second 200
#  - Third 100
# 3. Dropout of .4 at each layer
# 4. How many parameters does your model have?
# 4. Train this model for 20 epochs with RMSProp at a learning rate of .001 and a batch size of 128
# 
# 
# 

# In[ ]:




