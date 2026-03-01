
import os
import pathlib
import random
# random.seed(777)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.random import set_seed
set_seed(42)

os.chdir('/home/daniele/Ricerche/PhD_Genova/CNN')
work_dir = os.getcwd()

for dirpath, dirnames, filenames in os.walk('pizza_steak'):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
data_dir = pathlib.Path(f'{work_dir}/pizza_steak/train')    
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)    
    
def view_random_image(target_dir, target_class):
    target_folder = f'{target_dir}/{target_class}'
    
    random_image = random.sample(os.listdir(target_folder), 1)
    
    img = mpimg.imread(f'{target_folder}/{random_image[0]}')
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')
    plt.show()
    plt.close()
    
    print(f'Image shape: {img.shape}')
    
    return img

img = view_random_image(f'{work_dir}/pizza_steak/train', 'pizza')
    
# Modello
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
    
train_dir = 'pizza_steak/train'    
test_dir = 'pizza_steak/test'    

train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=32,
                                               # converti tutte le immagini in 244x244
                                               target_size=(224, 224),
                                               class_mode='binary',  # abbiamo solo due classi
                                               seed=42)

valid_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=32,
                                               # converti tutte le immagini in 244x244
                                               target_size=(224, 224),
                                               class_mode='binary',  # abbiamo solo due classi
                                               seed=42)

# Cosi non funziona. Perché?
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPool2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense

# model = Sequential() 
# model.add(Conv2D(filters=10, kernel_size=3, activation='relu',
#                  input_shape=(244, 244, 3)))
# model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
# model.add(MaxPool2D(pool_size=2, padding='valid'))
# model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
# model.add(Conv2D(filters=10, kernel_size=3, activation='relu'))
# model.add(MaxPool2D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(units=1, activation='sigmoid'))

import tensorflow as tf
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=10, 
                          kernel_size=3, # can also be (3, 3)
                          activation="relu", 
                          input_shape=(224, 224, 3)), # first layer specifies input shape (height, width, colour channels)
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=2, # pool_size can also be (2, 2)
                            padding="valid"), # padding can also be 'same'
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # activation='relu' == tf.keras.layers.Activations(tf.nn.relu)
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid") # binary activation output
])

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

history = model.fit(train_data, epochs=5,
                    steps_per_epoch=len(train_data),
                    validation_data=valid_data,
                    validation_steps=len(valid_data))

model.summary()

pd.DataFrame(history.history).plot(figsize=(10, 7))





































