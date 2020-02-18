from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, ZeroPadding2D
import matplotlib.pyplot as plt
from keras.layers import Dropout
import pandas as pd

def show_train_history(history, train, validation):
    plt.plot(history.history[train])
    plt.plot(history.history[validation])
    plt.title('Train History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# 資料處理
train_image = x_train_image.reshape(60000, 28, 28, 1).astype('float32') / 255.0
test_image = x_test_image.reshape(10000, 28, 28, 1).astype('float32') / 255.0

# train_image = x_train_image.astype('float32') / 255.0
# test_image = x_test_image.astype('float32') / 255.0

train_onehot = np_utils.to_categorical(y_train_label)
test_onehot = np_utils.to_categorical(y_test_label)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result = model.fit(train_image, train_onehot, validation_split=0.2, verbose=2, batch_size=100, epochs=20)

train_scores = model.evaluate(train_image, train_onehot)
print('Train accuracy:', train_scores[1])
test_scores = model.evaluate(test_image, test_onehot)
print('Test accuracy:', test_scores[1])

# show_train_history(result, 'accuracy', 'val_accuracy')

model.save('C:\\Users\\USER\\Desktop\\Keras\\model\\number_recognition.h5')