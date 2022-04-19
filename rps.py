import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

ds_name = 'rock_paper_scissors'

builder = tfds.builder(ds_name)

ds_train = tfds.load(name=ds_name, split='train')
ds_test = tfds.load(name=ds_name, split='test')


# data prep

train_images = np.array([example['image'].numpy()[:,:,0] for example in ds_train])
train_labels = np.array([example['label'].numpy() for example in ds_train])

test_images = np.array([example['image'].numpy()[:,:,0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])

train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255


# basic training
'''
model = keras.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(256, activation='relu'),
  keras.layers.Dense(3, activation='softmax')
])
'''

# convolutional approach
'''
model = keras.Sequential([
  keras.layers.Conv2D(64, 3, activation='relu', input_shape=(300,300,1)),
  keras.layers.Conv2D(32, 3, activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(3, activation='softmax')
])
'''

# better ConvNN

model = keras.Sequential([
  keras.layers.AveragePooling2D(6, 3, input_shape=(300,300,1)),
  keras.layers.Conv2D(64, 3, activation='relu'),
  keras.layers.Conv2D(32, 3, activation='relu'),
  keras.layers.MaxPool2D(2, 2),
  keras.layers.Dropout(0.5),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(3, activation='softmax')
])


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=32)

model.evaluate(test_images, test_labels)