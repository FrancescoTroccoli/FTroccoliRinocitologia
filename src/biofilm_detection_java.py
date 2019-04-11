import cv2
import os
import mahotas as mt
import numpy as np
import time
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from collections import Counter
from py4j.java_gateway import JavaGateway, GatewayParameters
from skimage import io


def haralick_features(src, filenames):
    """ Returns a dict containing 13 Haralick texture features for
    every image in the list of files given as input.

    Every image is firs converted to grayscale, only then Haralick features
    are extracted from the four 2D-directions, and the mean of the four directions values
    is calculated.

    Parameters
    ----------
    src : str
        The path which contains the files.
    filenames : list
        The list of files to elaborate.

    Returns
    -------
    features: dict of (str : list of np.double)
        The dictionary containing the list of features for every file.
    """
    features_list = dict()

    for file in filenames:
        image = cv2.imread(src + "/" + file)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = mt.features.haralick(grayscale_image)
        features = features.mean(axis=0)
        features_list[file] = features

    return features_list


def get_dominant_color(image, k=3):
    """Takes an image as input and returns the dominant color of the image as a list.

    Dominant color is found by running K-means on the
    pixels and returning the centroid of the largest cluster.

    Parameters
    ----------
    image : np.array-like image
        The image to elaborate.
    k : int, optional (default = 4)
        The number of clusters to form as well as the number of
        centroids to generate.

    Returns
    -------
    dominant_color : the h, s and v components of the dominant color
        in list form.
    """
    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)


def hsv_features(src, filenames, clusters=3):
    """Returns a list containing the HSV values of the dominant color of every image
    given as input.

    Parameters
    ----------
    src : str
        The path which contains the files.
    filenames : list
        The list of files to elaborate.
    clusters : int, optional (default = 3)
        The number of clusters to use for k-means elaboration.

    Returns
    -------
    dom_color_list : dict of (str : list of np.double)
        The dictionary containing the dominant color for every file.
    """
    dom_color_list = dict()

    for file in filenames:
        bgr_image = cv2.imread(src + "/" + file)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        dom_color = get_dominant_color(hsv_image, k=clusters)
        dom_color_list[file] = dom_color

    return dom_color_list


def train(src=None):
    """Trains the classifier and saves it on a file.

    This method elaborates the training dataset to extract haralick's and color features.

    The dataset is first normalized using Robust Scaler, then processed
    with PCA and finally used to train the classifier.

    The scaler, the PCA object and the classifier are saved on a file.

    Parameters
    ----------
    src : str, optional (default = "/training")
        The path to the directory which contains the training dataset
    """
    # Definition of the path
    if src is None:
        training_path = os.getcwd() + "/training"
    else:
        training_path = src

    # Create the list of the training files
    training_filenames = os.listdir(training_path)

    # Extract haralick features
    print("Extracting haralick features...")
    start_time = time.monotonic()
    haralick_training = haralick_features(training_path, training_filenames)
    end_time = time.monotonic()
    ex_time = timedelta(seconds=end_time - start_time)
    print("Completed in " + str(ex_time) + " seconds.")

    # Extract color features
    print("Extracting color features...")
    start_time = time.monotonic()
    hsv_training = hsv_features(training_path, training_filenames)
    end_time = time.monotonic()
    ex_time = timedelta(seconds=end_time - start_time)
    print("Completed in " + str(ex_time) + " seconds.")

    # Concatenate the lists to create a single feature vector per file
    training_set = dict()
    for file in training_filenames:
        training_set[file] = np.concatenate((haralick_training[file], hsv_training[file]))

    # Normalize the dataset with Robust Scaler and save the scaler on file
    print("Scaling the training set...")
    scaler = RobustScaler()
    scaler.fit(list(training_set.values()))
    joblib.dump(scaler, 'scaler.pkl')
    scaled_training = scaler.transform(list(training_set.values()))
    for file, scaled in zip(training_filenames, scaled_training):
        training_set[file] = scaled
    print("Scaling completed. Scaler saved on " + os.path.join(os.getcwd(), "scaler.pkl"))

    # Transform the dataset using PCA
    print("Started PCA processing...")
    pca = PCA()
    pca.fit(list(training_set.values()))
    print(os.path.join(os.getcwd(), "pca.pkl"))
    joblib.dump(pca, 'pca.pkl')
    pca_training = pca.transform(list(training_set.values()))
    for file, pca_value in zip(training_filenames, pca_training):
        training_set[file] = pca_value
    print("PCA transform complete. Fit PCA saved on " + os.path.join(os.getcwd(), "pca.pkl"))

    # Train the classifier
    print("Training the classifier...")
    if_clf = IsolationForest()
    if_clf.fit(list(training_set.values()))
    print("Classifier is ready!")

    # Save the classifier to file
    joblib.dump(if_clf, 'clf.pkl')
    print("Classifier saved on " + os.path.join(os.getcwd(), "clf.pkl"))


def predict_set(src=None):
    """Labels every image included in a given set.

    This method elaborates a set of images to extract haralick's and color features
    and uses those features to predict the presence or the absence of biofilm
    in the images.

    The feature vector is normalized using Robust Scaler, and then
    transformed using PCA.

    The used classifier is Isolation Forest.

    Both the scaler and the classifier are read from a file.

    Parameters
    ----------
    src : str, optional (default = "/test")
        The path to the directory which contains the images

    Returns
    -------
    labeled_set : dict of (str : str)
        The dictionary containing the labeled images.
    """
    # Definition of the path
    if src is None:
        test_path = os.getcwd() + "/test"
    else:
        test_path = src

    # Create the list of the test files
    test_filenames = os.listdir(test_path)

    # Extract haralick features from the list
    print("Extracting haralick features...")
    start_time = time.monotonic()
    haralick_test = haralick_features(test_path, test_filenames)
    end_time = time.monotonic()
    ex_time = timedelta(seconds=end_time - start_time)
    print("Completed in " + str(ex_time) + " seconds.")

    # Extract color features from the list
    print("Extracting color features...")
    start_time = time.monotonic()
    hsv_test = hsv_features(test_path, test_filenames)
    end_time = time.monotonic()
    ex_time = timedelta(seconds=end_time - start_time)
    print("Completed in " + str(ex_time) + " seconds.")

    # Concatenate the lists to create a single feature vector per file
    test_set = dict()
    for file in test_filenames:
        test_set[file] = np.concatenate((haralick_test[file], hsv_test[file]))

    # Load the scaler and normalize the dataset
    scaler = joblib.load('scaler.pkl')
    scaled_test = scaler.transform(list(test_set.values()))
    for file, scaled in zip(test_set.keys(), scaled_test):
        test_set[file] = scaled

    # Load the PCA and transform the dataset
    pca = joblib.load('pca.pkl')
    pca_test = pca.transform(list(test_set.values()))
    for file, pca_value in zip(test_set.keys(), pca_test):
        test_set[file] = pca_value

    # Load the classifier
    print("Loading classifier from file...")
    if_clf = joblib.load('clf.pkl')

    # Begin prediction
    labeled_set = dict()
    print("Labeling started.")
    for file in test_set.keys():
        prediction = if_clf.predict(test_set[file].reshape(1, -1))
        if prediction == 1:
            prediction = "BIOFILM"
        else:
            prediction = "OTHER"

        labeled_set[file] = prediction

    return labeled_set


def predict_one(image, scaler, pca, clf):
    """ Predicts the presence or the absence of biofilm in a single image.

    Parameters
    ----------
    image : np.array
        The image to elaborate.
    scaler : a sklearn.preprocessing scaler
        The scaler used to normalize the feature vector.
    pca : a sklearn.decomposition.PCA object
        The trained PCA used to compute the orthogonal transformation over the dataset
    clf: the classifier.

    Returns
    -------
    prediction : str
        The predicted label for the image ["BIOFILM", "OTHER"]
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mt.features.haralick(grayscale_image)
    haralick = haralick.mean(axis=0)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dom_color = get_dominant_color(hsv_image)

    feature_vector = np.concatenate((haralick, dom_color))

    scaled_vector = scaler.transform(feature_vector.reshape(1, -1))

    pca_vector = pca.transform(scaled_vector.reshape(1, -1))

    prediction = clf.predict(pca_vector.reshape(1, -1))
    if prediction == 1:
        prediction = "BIOFILM"
    else:
        prediction = "OTHER"

    return prediction


if __name__ == '__main__':

    # connect to the JVM
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25335))

    # get the patient path.
    patient = gateway.entry_point
    load_path = patient.getPathCampi()

    # get the results path
    biofilm_path = patient.getPathBiofilmsi()
    other_path = patient.getPathBiofilmno()

    # load the scaler, the PCA and the classifier
    scaler = joblib.load('scaler.pkl')
    clf = joblib.load('clf.pkl')
    pca = joblib.load('pca.pkl')

    for file in os.listdir(load_path):
        # if file.endswith(('.jpg', '.png', '.JPG', '.PNG')):
        if file.endswith(('.jpg', '.JPG')):
            print(file)
            image = cv2.imread(load_path + "/" + file)
            height, width = image.shape[:2]
            step = 128
            height = height - (height % step)
            c = 0
            for y in range(0, height, step):
                for x in range(0, width, step):
                    crop = image[y:y + step, x:x + step]
                    label = predict_one(crop, scaler, pca, clf)

                    if label == "BIOFILM":
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        io.imsave(biofilm_path + "/" + str(c) + "-" + file, crop)
                    else:
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        io.imsave(other_path + "/" + str(c) + "-" + file, crop)
                    c += 1
