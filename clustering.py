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
    model = DBSCAN(eps=1.2*avg_width, min_samples=2)
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
# load image first
img = cv2.imread(img_name)
# now get variables passing in the image NOT just the string of the filename
cups, avg_width, avg_height = get_locations(img)
# define the model
model = DBSCAN(eps=1.2*avg_width, min_samples=2)
# fit the model
model.fit(cups)
# assign a cluster to each example
yhat = model.fit_predict(cups)

# print(cups[:,0])
# no water cup
if len(cups) != 11:
    print("Error: {}/11 cups detected".format(len(cups)))

x = cups[:, 0]
y = cups[:, 1]
mag , ang = (np.zeros((len(x), len(x))), )*2
print(mag)
for i in range(0, len(cups)):
    for j in range(0, len(cups)):
        x_diff = x[i]-x[j]
        y_diff = y[i]-y[j]
        mag_ij = (x_diff**2+y_diff**2)**0.5
        if (x[i]-x[j]) == 0:
            ang[i,j] = 0
        else:
            ang[i, j] = np.arctan((x[i]-y[j])/(x[i]-x[j]))*(180/np.pi)

print(mag)
#print(mag.min(), mag.max(), mag)

for i in range(0, len(cups)):
    for j in range(0, len(cups)):
        if mag[i, j] <= 10:
           poo = 0



# no hits

# hits only on left

# hits only on right

# hits on both

plt.show()
