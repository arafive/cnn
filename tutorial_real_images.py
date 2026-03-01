
'''
Descrizione: tutorial CNN

             Malaria Dataset

'''

import os
import random
random.seed(777)
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

from matplotlib.image import imread

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 8}

matplotlib.rc('font', **font)

os.chdir('/home/daniele/Ricerche/PhD_Genova/CNN')
work_dir = os.getcwd()

#%% Preprocessing

train_para_path = f'{work_dir}/cell_images/train/parasitized'
train_uninf_path = f'{work_dir}/cell_images/train/uninfected'
test_para_path = f'{work_dir}/cell_images/test/parasitized'
test_uninf_path = f'{work_dir}/cell_images/test/uninfected'

train_para = [x for x in os.listdir(train_para_path) if 'png' in x]
train_uninf = [x for x in os.listdir(train_uninf_path) if 'png' in x]
test_para = [x for x in os.listdir(test_para_path) if 'png' in x]
test_uninf = [x for x in os.listdir(test_uninf_path) if 'png' in x]

# para_cell = f'{train_para_path}/{train[0]}'
para_cell = f'{train_para_path}/C100P61ThinF_IMG_20150918_144104_cell_162.png'
imread(para_cell)
imread(para_cell).shape

plt.imshow(imread(para_cell))
plt.axis('off')
plt.show()
plt.close()

uninf_cell = f'{train_uninf_path}/C100P61ThinF_IMG_20150918_144104_cell_128.png'
imread(uninf_cell)

plt.imshow(imread(uninf_cell))
plt.axis('off')
plt.show()
plt.close()

# Bisogna riportare tutte le immagini alla stessa shape

dim1, dim2 = [], []

for i in os.listdir(test_uninf_path):
    img = imread(f'{test_uninf_path}/{i}')
    d1, d2, colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

# Dobbiamo decidere quali dimensioni dare a tutte le immagini
np.mean(dim1)
np.mean(dim2)

image_shape = (130, 130, 3)

image_gen = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               # rescale nel caso non fossero normalizzate
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')

plt.imshow(imread(para_cell))
plt.axis('off')
plt.show()
plt.close()

plt.imshow(image_gen.random_transform(imread(para_cell))) # potrei espandere a dismisura il mio dataset
plt.axis('off')
plt.show()
plt.close()

image_gen.flow_from_directory(f'{work_dir}/cell_images/train')
# le cartelle devono essere in uno specifico ordine
# ogni sub-directory deve contenre una classe

image_gen.flow_from_directory(f'{work_dir}/cell_images/test')

#%% Model

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid')) # sigmoid perché abbiamo solo due classi

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=2)

batch_size = 16

train_image_gen = image_gen.flow_from_directory(f'{work_dir}/cell_images/train',
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory(f'{work_dir}/cell_images/test',
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                shuffle=False)

train_image_gen.class_indices
test_image_gen.class_indices

model.fit(train_image_gen, epochs=20, validation_data=test_image_gen, callbacks=[early_stop])

model.metrics_names
model.evaluate(test_image_gen) # nel tutorial il modello lavora un po' peggio...

pred = model.predict(test_image_gen)
# predictions = pred > 0.5
predictions = pred > 0.8

print(classification_report(test_image_gen.classes, predictions))

print(confusion_matrix(test_image_gen.classes, predictions))

my_img_test = image.load_img(para_cell, target_size=image_shape)
my_img_test

my_img_arr = image.img_to_array(my_img_test)
my_img_arr.shape

model.predict(my_img_arr)
