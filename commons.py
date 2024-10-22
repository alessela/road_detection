import cv2
import numpy as np

class WeakLearner:
    def __init__(self, feature_i, threshold, class_label, error) -> None:
        self.feature_i = feature_i
        self.threshold = threshold
        self.class_label = class_label
        self.error = error

    def classify(self, x):
        return self.class_label if x[self.feature_i] < self.threshold else -self.class_label

    def __str__(self) -> str:
        return f'feature_i = {self.feature_i}, threshold = {self.threshold}, class_label = {self.class_label}, error = {self.error}'

class StrongLearner:
    def __init__(self, hs: list[tuple[WeakLearner, float]], threshold=0) -> None:
        self.hs = hs
        self.threshold = threshold

    def classify(self,x):
        return 1 if sum([alpha * wl.classify(x) for wl, alpha in self.hs]) > self.threshold else -1

    def error(self):
        return sum([alpha * wl.error for wl, alpha in self.hs]) / len(self.hs)

imagesPath = 'data_road/training/image_2'
groundTruthPath = 'data_road/training/gt_image_2'

def local_binary_patterns(img: cv2.Mat):
    rows, cols = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows-1):
        for j in range(cols-1):
            if gray[i, j+1] > gray[i, j]:
                lbp[i, j] += 1
            elif gray[i, j+1] < gray[i, j]:
                lbp[i, j+1] += 4
            else:
                lbp[i, j] += 1
                lbp[i, j+1] += 4
            
            if gray[i+1, j] > gray[i, j]:
                lbp[i, j] += 2
            elif gray[i+1, j] < gray[i, j]:
                lbp[i+1, j] += 8
            else:
                lbp[i, j] += 2
                lbp[i+1, j] += 8

    for i in range(rows-1):
        if gray[i+1, cols-1] > gray[i, cols-1]:
            lbp[i, cols-1] += 2
        elif gray[i+1, cols-1] < gray[i, cols-1]:
            lbp[i+1, cols-1] += 8
        else:
            lbp[i, cols-1] += 2
            lbp[i+1, cols-1] += 8

    for j in range(cols-1):
        if gray[rows-1, j+1] > gray[rows-1, j]:
            lbp[rows-1, j] += 1
        elif gray[rows-1, j+1] < gray[rows-1, j]:
            lbp[rows-1, j+1] += 4
        else:
            lbp[rows-1, j] += 1
            lbp[rows-1, j+1] += 4
    return lbp

def hist(img):
    h = [0] * 16
    for line in img:
        for pixel in line:
            h[pixel] += 1
    return h

def is_inside(img: cv2.Mat, x, y):
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]