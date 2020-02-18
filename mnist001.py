from keras.utils import np_utils
import numpy as np
np.random.seed(10)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
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

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25

    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx], cmap = 'binary')

        title = "label = " + str(labels[idx])
        if len(prediction) > 0:
            title += ", predict = " + str(prediction[idx])

        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

# 讀取mnist資料
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# 資料處理
train_image = x_train_image.reshape(60000, 784).astype('float32')
test_image = x_test_image.reshape(10000, 784).astype('float32')

train_image /= 255
test_image /= 255

train_onehot = np_utils.to_categorical(y_train_label)
test_onehot = np_utils.to_categorical(y_test_label)


model = Sequential()
model.add(Dense(input_dim=784, units=1000, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
result = model.fit(train_image, train_onehot, validation_split=0.2, verbose=2, batch_size=1000, epochs=20)

train_scores = model.evaluate(train_image, train_onehot)
print('Train accuracy:', train_scores[1])
test_scores = model.evaluate(test_image, test_onehot)
print('Test accuracy:', test_scores[1])

predict = model.predict_classes(test_image)
plot_images_labels_prediction(x_test_image, y_test_label, predict, idx=340, num=25)

show_train_history(result, 'accuracy', 'val_accuracy')

# 顯示混淆矩陣
# print(pd.crosstab(y_test_label, predict, rownames=['original'], colnames=['prediction']))