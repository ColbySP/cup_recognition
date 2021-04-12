# imports
import cv2
import numpy as np
import cvlib as cv
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def get_locations(image):
    bbox, label, conf = cv.detect_common_objects(image=image, confidence=0.5)

    if 'cup' in label:
        indices = [i for i, x in enumerate(label) if x == "cup"]
        bbox = [bbox[i] for i in indices]

    cups = []
    cup_width, cup_height = [], []
    for cup in bbox:
        x_avg = (cup[0] + cup[2]) / 2
        y_avg = (cup[1] + cup[3]) / 2
        cup_width.append(cup[2] - cup[0])
        cup_height.append(cup[3] - cup[1])
        cups.append([x_avg, y_avg])

    return np.array(cups), abs(np.mean(cup_width)), abs(np.mean(cup_height))


def dbscan_clustering(file_name):
    # load image
    img = cv2.imread(file_name)
    # define dataset
    cups, avg_width, avg_height = get_locations(img)
    # define the model
    model = DBSCAN(eps=2 * avg_width, min_samples=2)
    # fit the model
    model.fit(cups)
    # assign a cluster to each example
    yhat = model.fit_predict(cups)
    # retrieve unique clusters
    clusters = unique(yhat)
    # initialize a figure
    plt.figure(1)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(cups[row_ix, 0], cups[row_ix, 1])
    # title the plot
    plt.title('DBSCAN Grouping Of Cups')
    # show the plot
    plt.imshow(img)


img_name = 'pong2.jpg'
dbscan_clustering(img_name)
plt.show()
