# Import libraries and modules

import os

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


def gen_images(img):
    """
    Generate images according to the ImageDataGenerator options
    :param img: image
    """
    i = 0
    for batch in datagen.flow(img, save_to_dir='images_generated', save_prefix='cell', save_format='png'):
        i += 1
        if i > 9:
            break


# Image agumentation options
datagen = ImageDataGenerator(
    rotation_range=50.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect'
)

# Create directory to save new images
if not os.path.exists(os.getcwd() + "/images_generated"):
    os.makedirs("images_generated/")

# Load image and generate new images
for path, dirs, files in os.walk(os.getcwd() + "/RUS/0"):
    for f in files:
        if f.endswith('.PNG') or f.endswith('.png'):
            img = load_img(os.path.join(path, f))
            x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 150, 150)
            gen_images(x)

