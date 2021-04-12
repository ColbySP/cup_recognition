import cv2
import cvlib as cv
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox


def detect_cups(image):
    # load the image
    im = cv2.imread(image)
    # detect common objects and their attributes
    bbox, label, conf = cv.detect_common_objects(im)
    # specify only cups
    if 'cup' in label:
        indices = [i for i, x in enumerate(label) if x == "cup"]
        label = [label[i] for i in indices]
        bbox = [bbox[i] for i in indices]
        conf = [conf[i] for i in indices]
    # create an output image
    output_image = draw_bbox(im, bbox, label, conf)

    return output_image, bbox, label, conf


output_image, boxes, labels, confidence = detect_cups('test1.jpg')

xs = []
ys = []
for elem in boxes:
    x = elem[2] - elem[0]
    y = elem[3] - elem[1]
    xs.append(x)
    ys.append(y)

print(xs)
print(ys)

plt.imshow(output_image)
plt.show()
