#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
print(tf.__version__)


# In[56]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[49]:


# T-shirt/top',= 0
# 'Trouser', = 1
# 'Pullover', = 2
# 'Dress', = 3
# 'Coat', = 4
# 'Sandal',= 5
# 'Shirt',= 6
# 'Sneaker',= 7
# 'Bag',= 8
# 'Ankle boot'= 9


# In[57]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[58]:


# Explore the data

train_images.shape


# In[6]:


train_labels.shape


# In[7]:


test_images.shape


# In[8]:


test_labels.shape


# In[9]:


len(train_labels)


# In[10]:


# Each label is an integer between o and 1

train_labels


# In[11]:


len(test_labels)


# In[12]:


# Prepocess the data


# In[13]:


# The data must be preprocessed before training the network. 
# If you inspect the first image in the training set, you will see that the pixel values 
# fall in the range of 0 to 255:

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[14]:


# We scale these values to a range of 0 to 1 before feeding to the neural network model.
# For this, we divide the values by 255. It's important that the training set and the testing 
# set are preprocessed in the same way:

train_images = train_images / 255.0

test_images = test_images / 255.0


# In[15]:


# Display the first 25 images from the training set and display the class name below 
# each image. Verify that the data is in the correct format and we're ready to build and train 
# the network.


# In[16]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[17]:


# Build the model
# Building the neural network requires configuring the layers of the model, then compiling 
# the model.


# In[18]:


# Set up the layers

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])


# In[19]:


# Compile the model
# Loss function
# Optimizer
# Metrics

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


# Train the model

# Training the neural network model requires the following steps:

# 1. Feed the training data to the model—in this example, the train_images and train_labels 
#    arrays.
# 2. The model learns to associate images and labels.
# 3. We ask the model to make predictions about a test set—in this example, 
#    the test_images array. We verify that the predictions match the labels from the 
#    test_labels array


# In[21]:


model.fit(train_images, train_labels, epochs=5)


# In[22]:


# Evaluate accuracy

# Next, compare how the model performs on the test dataset:

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[25]:


# Make predicitons

# With the model trained, we can use it to make predictions about some images.

predictions = model.predict(test_images)


# In[27]:


# Take a look at the first prediction

predictions[0]


# In[28]:


np.argmax(predictions[0])


# In[31]:


# Verify with tha label of prediciton 0

test_labels[0]


# In[32]:


# We can graph this to look at the full set of 10 channels


# In[33]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[34]:


# Let's look at the 0th image, predictions, and prediction array.


# In[35]:


i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[36]:


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[37]:


i = 5000
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[38]:


i = 1354
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[39]:


# Correct prediction labels are blue and incorrect prediction labels are red. 
# The number gives the percent (out of 100) for the predicted label. 
# Note that it can be wrong even when very confident.


# In[40]:


# Finally, use the trained model to make a prediction about a single image
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)


# In[41]:


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


# In[42]:


# Now predict the image:

predictions_single = model.predict(img)

print(predictions_single)


# In[45]:


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


# In[46]:


# model.predict returns a list of lists, one for each image in the batch of data. 
# Grab the predictions for our (only) image in the batch:


# In[47]:


np.argmax(predictions_single[0])


# In[ ]:


# And, as before, the model predicts a label of 9

