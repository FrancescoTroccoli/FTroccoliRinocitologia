# Import libraries and modules
import itertools
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.regularizers import l1
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state in order to obtain reproducible results
np.random.seed(42)

# Number of classes
num_class = 8


def load_data(data_directory):
    """
    Loading images and labels from directory
    :param data_directory:
    :return: images, labels
    """
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith('.png') or f.endswith('.PNG')]
        for f in file_names:
            print(f != 0)
            img = cv2.imread(f)
            (b, g, r) = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img, (50, 50))
            images.append(img)
            labels.append(int(d))
    return images, labels


def create_model(input_shape):
    """
    Create CNN model
    :param input_shape: shape image
    :return: model
    """

    # Create Sequential model
    model = Sequential()

    # Add Conv-layer and initialize it
    model.add(Conv2D(32, (5, 5), padding='same',
                     input_shape=input_shape, kernel_initializer='random_uniform', bias_initializer='zeros'
                     ))
    # Add Activation-layer to introduce non-linearity
    model.add(Activation('relu'))
    # Add MaxPool-layer to downsample the feature-map
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add Dropout for regularization
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Convert the 3D feature maps to 1D feature vectors
    model.add(Flatten())

    # Fully connected layer with 512 neurons
    model.add(Dense(512))

    # Add batch normalization to speed up the training
    model.add(BatchNormalization())

    model.add(Dropout(0.73))

    # Add last layer with 8 neurons
    model.add(Dense(num_class))

    # Add softmax classifier
    model.add(Activation('softmax'))

    return model



#def get_featuremaps(model, layer_idx, X_batch):
    #get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output, ])
    #activations = get_activations([X_batch, 0])
    #return activations


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Load dataset
ROOT_PATH = os.getcwd() + "/"
data_directory = os.path.join(ROOT_PATH, "Dataset/")
xds_list, yds_list = load_data(data_directory)

# Preprocessing data
xds_list = np.array(xds_list)
xds_list = xds_list.astype("float32")
xds_list /= 255
xds_list = xds_list.reshape(xds_list.shape[0], 50, 50, 3)
yds_list = np.array(yds_list)

# Convert 1-dimensional class arrays to 8-dimensional class matrices
yds_list_cat = np_utils.to_categorical(yds_list, num_class)

# Shuffle data
x, y = shuffle(xds_list, yds_list_cat, random_state=2)

# Split into training ad testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Set input shape
input_shape = x_train[0].shape

# Create model
model = create_model(input_shape)

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Save model created
model.save('my_model.h5')

# Set filepath to save best weights of the model
filepath = "weights.best.hdf5"

# Set checkpoint to save the best weights
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#es = EarlyStopping(monitor="val_acc", mode="max", verbose=1, patience=30)
callbacks_list = [checkpoint]

# Train the model
hist = model.fit(x, y, batch_size=64, epochs=60,
                 validation_split=0.1, verbose=2,
                 callbacks=callbacks_list)

# Plot train and validation loss then train and validation accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(10)

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(hist.history['loss'], 'r', linewidth=3.0)
plt.plot(hist.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)
plt.show()

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(hist.history['acc'], 'r', linewidth=3.0)
plt.plot(hist.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()

# Load the best weights
model.load_weights(filepath)

# Evaluating the model
score = model.evaluate(x_test, y_test, verbose=0)

print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# Plotting the confusion matrix

Y_pred = model.predict(x_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)

target_names = ['class 0(Epiteliali)', 'class 1(Neutrofili)', 'class 2(Eosinofili)','class 3(Mastcellule)',
                'class 4(Linfociti)', 'class 5(Mucipare)', 'class 6(Macchie)', 'class 7(Nuclei_nudi)']

print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test, axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')

plt.show()
