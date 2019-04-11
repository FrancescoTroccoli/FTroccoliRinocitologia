# Import libraries and modules
import os
from os import path
import numpy as np
import shutil
import configparser
import skimage
from src.data_manager import DataManager
from keras.models import load_model

INPUT_IMAGE_WIDTH = 50
INPUT_IMAGE_HEIGHT = 50


def load_resize_img(img_path):
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float32(img)

    return skimage.transform.resize(img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))


def load_data(cell_pathectory):

    """
    Loading images and labels from cell directory
    :param cell_directory: directory path with extracted cell images
    :return: images, image'names
    """

    file_names = data_manager.get_cells_images()
    images = [load_resize_img(img_name) for img_name in file_names]

    return images, file_names


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_manager = DataManager.from_file()

    # load the trained model.
    model = load_model(config["Models"]["classifier"])
    model.load_weights(config["Models"]["classifier_weights"])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # load cell images
    images, file_names = load_data(data_manager.get_cells_path())

    # images to multi dimensional arrays
    images = np.array(images)

    out_path = data_manager.get_output_path()

    for i, (img, img_name) in enumerate(zip(images, file_names)):
        img = img.reshape((1,)+img.shape)
        # add a dimension to the image
        img_class = model.predict_classes(img)

        # define the path of its folder-class (classify the image).
        class_path = data_manager.get_cell_class_path(img_class[0])

        # move it to the correct destination.
        shutil.move(img_name, class_path)
        print("Predict class for image {}: {}".format(file_names[i], class_path))
