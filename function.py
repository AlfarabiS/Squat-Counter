import numpy as np
import cv2


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        return 360 - angle
    else:
        return angle


def classifier(hip_angle, knee_angle):
    classname = None
    if hip_angle < 85:
        if knee_angle < 130:
            classname = 'perfect'
    elif hip_angle < 70:
        if knee_angle < 135:
            classname = 'hip terlalu menekuk'
    return classname


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def counter(hip_angle, knee_angle, init_reps):
    counter = init_reps
    stage = None
    if knee_angle > 150 or hip_angle > 150:
        stage = "up"
    elif knee_angle <= 90 or hip_angle < 90 and stage == 'up':
        stage = "down"
        counter = counter + 1
    return counter
