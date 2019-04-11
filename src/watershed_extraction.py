# ---------------------------------------------------
# Cells extraction using the watershed method from
# OpenCv library
# ---------------------------------------------------

import os
import time
import logging
import configparser
from datetime import timedelta

import numpy as np
from scipy import ndimage
import cv2

from skimage import morphology
from skimage import io
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from src.data_manager import DataManager
from collections import namedtuple
import matplotlib as mpl
mpl.cm
LOW_THRESHOLD_SIZE = 1000
HIGH_THRESHOLD_SIZE = 15000


def detect_cells(field_image, return_steps=False):

    extraction_steps = namedtuple("ExtractionSteps", ["input", "meanshift",
                                                      "grayscale", "binary",
                                                      "dilation", "distance",
                                                      "markers", "labels",
                                                      "filtered_labels"
                                                      ])



    #selection of RGB's blue channel
    channel_r=field_image[..., 0]
    #thresholding otsu to separate cells from background

    # perform pyramid mean shift filtering
    # to aid the thresholding step
    shifted = cv2.pyrMeanShiftFiltering(field_image, 21, 51)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    # gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # morphological transformation
    selem = morphology.disk(5)
    dilated = morphology.dilation(binary, selem)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    dist_map = ndimage.distance_transform_edt(dilated)
    local_max = peak_local_max(dist_map, indices=False, min_distance=20,
                               labels=dilated)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = morphology.watershed(-dist_map, markers, mask=dilated)
    
    # Remove labels too small and too big
    filtered_labels = np.copy(labels)
    component_sizes = np.bincount(labels.ravel())

    too_small = component_sizes < LOW_THRESHOLD_SIZE
    too_small_mask = too_small[labels]
    filtered_labels[too_small_mask] = 1

    too_big = component_sizes > HIGH_THRESHOLD_SIZE
    too_big_mask = too_big[labels]
    filtered_labels[too_big_mask] = 1

    if return_steps:
        return extraction_steps(input=field_image, meanshift=shifted,
                                grayscale=gray, binary=binary,
                                dilation=dilated, distance=dist_map,
                                markers=markers, labels=labels,
                                filtered_labels=filtered_labels)
    return filtered_labels


def extract_cells(cell_labels, image_index, out_path):

    regions = regionprops(cell_labels)
    export_img_extension = data_manager.get_output_extension()

    for i, region in enumerate(regions[1:]):  # jump the first region (regions[0]) because is the entire image

        #y0, x0 = region.centroid

        #print("Cell n°", i, ": y=", y0, "x x=", x0)

        minr, minc, maxr, maxc = region.bbox

        # Transform the region to crop from rectangular to square
        x_side = maxc - minc
        y_side = maxr - minr
        if x_side > y_side:
            maxr = x_side + minr
        else:
            maxc = y_side + minc

        if (minc > 20) & (minr > 20):
            minc = minc - 20
            minr = minr - 20

        cell = image[minr:maxr + 20, minc:maxc + 20]  # crop image

        # save the image
        img_name = "img#" + str(image_index) + "_cell#" + str(i) + export_img_extension
        filepath = os.path.join(out_path, img_name)
        io.imsave(filepath, cell)

        logging.info("extracted cell: {}".format(img_name))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s")
    config = configparser.ConfigParser()
    config.read("config.ini")
    data_manager = DataManager.from_file()
    start_time = time.monotonic()

    inpath = data_manager.get_input_path()
    outpath = data_manager.get_cells_path()

    logging.info("input path: {}".format(inpath))
    logging.info("extracted cells will be saved in: {}".format(outpath))

    files = data_manager.get_input_images()

    if not files:
        logging.error("{} directory is empty! No image to process".format(inpath))

    for i, infile in enumerate(files):
        logging.info("processing {} image".format(infile))

        image = cv2.imread(infile)

        # transform the color scheme to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            labels = detect_cells(image)
            extract_cells(labels, i, outpath)
        except ValueError:
            continue

    end_time = time.monotonic()
    logging.info("cells extraction time: {}".format(timedelta(seconds=end_time - start_time)))
