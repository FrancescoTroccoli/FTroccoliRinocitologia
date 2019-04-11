#coding=utf-8

from __future__ import print_function

try:
    import os
except:
    pass

try:
    import cv2
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import uniform, choice
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    from sklearn.utils import shuffle
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

ROOT_PATH = os.getcwd() + "/"

directories = [d for d in os.listdir(ROOT_PATH + "Dataset/")
               if os.path.isdir(os.path.join(ROOT_PATH + "Dataset/", d))]
yds_list = []
xds_list = []

for d in directories:
    label_directory = os.path.join(ROOT_PATH + "Dataset/", d)
    file_names = [os.path.join(label_directory, f)
                  for f in os.listdir(label_directory)
                  if f.endswith('.png') or f.endswith('.PNG')]
    for f in file_names:
        img = cv2.imread(f)
        (b, g, r) = cv2.split(img)
        img = cv2.merge([r, g, b])
        img = cv2.resize(img, (25, 25))
        xds_list.append(img)
        yds_list.append(int(d))
# Preprocessing data
xds_list = np.array(xds_list)
xds_list = xds_list.astype("float32")
xds_list /= 255
xds_list = xds_list.reshape(xds_list.shape[0], 25, 25, 3)
yds_list = np.array(yds_list)

# Convert 1-dimensional class arrays to 7-dimensional class matrices
yds_list_cat = np_utils.to_categorical(yds_list, 8)

# Shuffle data
x, y = shuffle(xds_list, yds_list_cat, random_state=2)

# Split into training ad testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



def keras_fmin_fnct(space):

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train[0].shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(space['Dropout']))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(space['Dropout_1']))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(space['Dropout_2']))

    model.add(Flatten())
    model.add(Dense(space['Dense']))
    model.add(Dropout(space['Dropout_3']))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=space['optimizer'])
    model.fit(x_train, y_train,
              batch_size=space['batch_size'],
              epochs=space['epochs'],
              verbose=2,
              validation_data=(x_test, y_test))

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Dropout': hp.uniform('Dropout', 0, 1),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'Dropout_2': hp.uniform('Dropout_2', 0, 1),
        'Dense': hp.choice('Dense', [256, 512, 1024]),
        'Dropout_3': hp.uniform('Dropout_3', 0, 1),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'batch_size': hp.choice('batch_size', [64, 128]),
        'epochs': hp.choice('epochs', [20, 50, 60]),
    }
