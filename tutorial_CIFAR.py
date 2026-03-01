
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

from tensorflow.keras.datasets import cifar10
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

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# batch_size, width, height, channel=3 (RGB)
X_train.shape

X_train[0].shape

plt.imshow(X_train[12])
plt.axis('off')
plt.show()
plt.close()

X_train = X_train/255 # si applica a tutti e 3 i canali
X_test = X_test/255

y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)

#%% Model and Training

model = Sequential()

# CONV 1
model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(32,32,3), activation='relu'))
# POOL 1
model.add(MaxPool2D(pool_size=(2,2)))

# CONV 2
model.add(Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='valid', input_shape=(32,32,3), activation='relu'))
# POOL 2
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(units=256, activation='relu'))

model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

model.fit(X_train, y_cat_train, epochs=10, validation_data=(X_test, y_cat_test), callbacks=[early_stop], verbose=1)

#%% Model evaluation
metrics = pd.DataFrame(model.history.history)
metrics.columns

metrics[['accuracy', 'val_accuracy']].plot()
metrics[['loss', 'val_loss']].plot()

model.evaluate(X_test, y_cat_test, verbose=2)

predictions = np.argmax(model.predict(X_test),axis=1)
print(classification_report(y_test, predictions))

plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True)

my_image = X_test[0] # 0 è un gatto

np.argmax(model.predict(my_image.reshape(1,32,32,3)))
