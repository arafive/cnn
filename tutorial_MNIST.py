
'''
Descrizione: tutorial CNN

'''

import os
import random
random.seed(777)
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

font = {'family': 'Helvetica',
        'weight': 'normal',
        'size': 8}

matplotlib.rc('font', **font)

os.chdir('/home/daniele/Ricerche/PhD_Genova/CNN')
work_dir = os.getcwd()

#%% Preprocessing

# Con il dataset MNIST la nostra shape sarà:
# 60.000x28x28x1, dove 1 è l'unico canale

# Se i numeri fossero stati a calori avremmo avuto 3 come ultima dimensione

# I label li trasformeremo in on-hot encoding. Ad esempio se abbiamo un 4
# avremo [0,0,0,1,0,0,0,0,0] e non [4]. Facciamo così perché è più
# comodo con un layer di uscita con 10 neuroni di output

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape
single_image = X_train[0]

plt.imshow(single_image, cmap='Greys')
plt.axis('off')
plt.show()
plt.close()

y_cat_train = to_categorical(y_train) # ora y_train è one-hot encoded
y_cat_test = to_categorical(y_test)

# Normalizzazione dei dati
X_train = X_train/255
X_test = X_test/255

scaled_image = X_train[0]

# Reshape
X_train.shape
# Dobbiamo aggiungere una dimensione, cioè il canale
# batch_size, width, height, channel
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#%% Model and Training

model = Sequential()

### Rules-of-Thumb
# Più complessi sono i dati, più filtri serviranno

### Opzione padding
# VALID: Don't apply any padding, i.e., assume that all dimensions are valid so
# that input image fully gets covered by filter and stride you specified.

# SAME: Apply padding to input (if needed) so that input image gets fully
# covered by filter and stride you specified. For stride 1, this will ensure
# that output image size is same as input.

# Gli hyperparamers con cui si può giocare sono:
    # activation
    # units
    # numero di layer
    # kernel_size
    # numero di filti
    # pool_size
    # padding

model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))

# Output layer Softmax --> Multi class
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

model.fit(X_train, y_cat_train, epochs=10, validation_data=(X_test, y_cat_test), callbacks=[early_stop], verbose=1)

#%% Model evaluation

metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()

# il loss viene dalla loss scelta
# l'accuracy è quante volte predici bene su tutti i predict

print(model.metrics_names)
model.evaluate(X_test, y_cat_test, verbose=2)

predictions = np.argmax(model.predict(X_test),axis=1)
print(classification_report(y_test, predictions))

plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)
## classi previste
# c
# l
# a
# s
# s
# i
#
# v
# e
# r
# e

my_number = X_test[0]
plt.imshow(my_number.reshape(28,28), cmap='Greys')
plt.axis('off')
plt.show()
plt.close()

np.argmax(model.predict(my_number.reshape(1,28,28,1)))
