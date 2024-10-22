from math import exp, inf, log, pi
from os import listdir
from random import randrange
from time import time
import cv2
import numpy as np

def sign(x):
    return 1 if x > 0 else -1

def isInside(img,x,y):
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]

class WeakLearner:
    def __init__(self, feature_i, threshold, class_label, error) -> None:
        self.feature_i = feature_i
        self.threshold = threshold
        self.class_label = class_label
        self.error = error
        self.errors = []

    def classify(self, x):
        return self.class_label if x[self.feature_i] < self.threshold else -self.class_label

    def __str__(self) -> str:
        return f'feature_i = {self.feature_i}, threshold = {self.threshold}, class_label = {self.class_label}, error = {self.error}'

class StrongLearner:
    def __init__(self, hs) -> None:
        self.hs = hs

    def classify(self,x):
        return sign(sum([alpha * wl.classify(x) for wl, alpha in self.hs]))

    def error(self):
        return sum([alpha * wl.error for wl, alpha in self.hs]) / len(self.hs)

def findWeakLearner(wls,w):
    best_h = WeakLearner(0,0,0,inf)

    for feature in wls:
        for cls in feature:
            for h in cls:
                h.error = sum([w[i] for i in h.errors])
                if h.error < best_h.error:
                    best_h = h

    return best_h

def adaboost(trainSet, groundTruth, T, k=100):
    x, y = [], []
    start = time()

    for file, gt_file in zip(listdir(trainSet), listdir(groundTruth)):
        img = cv2.imread(trainSet + '/' + file, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gt_img = cv2.imread(groundTruth + '/' + gt_file, 1)

        rows, cols = img.shape[0], img.shape[1]

        set_road = set()
        set_nonroad = set()

        while len(set_road) < k or len(set_nonroad) < k:
            i, j = randrange(rows), randrange(cols)
            if np.array_equal(gt_img[i,j], [255,0,255]):
                if len(set_road) < k:
                    set_road.add((i,j))
            else:
                if len(set_nonroad) < k:
                    set_nonroad.add((i,j))

        for i, j in list(set_road) + list(set_nonroad):
            npatch = 25
            s_hsv0, s_hsv1, s_hsv2 = 0, 0, 0

            for pi in range(i - 2, i + 3):
                for pj in range (j - 2, j + 3):
                    if isInside(img, pi, pj):
                        s_hsv0 += int(hsv[pi,pj,0])
                        s_hsv1 += int(hsv[pi,pj,1])
                        s_hsv2 += int(hsv[pi,pj,2])
                    else:
                        npatch -= 1

            s_hsv0 //= npatch
            s_hsv1 //= npatch
            s_hsv2 //= npatch
            x.append((i, j, s_hsv0, s_hsv1, s_hsv2))

        y.extend([1] * k + [-1] * k)

    wls = [[[WeakLearner(0, threshold, class_label, 0) for threshold in range(rows)] for class_label in [-1, 1]],
            [[WeakLearner(1, threshold, class_label, 0) for threshold in range(cols)] for class_label in [-1, 1]],
            [[WeakLearner(2, threshold, class_label, 0) for threshold in range(256)] for class_label in [-1, 1]],
            [[WeakLearner(3, threshold, class_label, 0) for threshold in range(256)] for class_label in [-1, 1]],
            [[WeakLearner(4, threshold, class_label, 0) for threshold in range(256)] for class_label in [-1, 1]]]

    for feature in wls:
        for cls in feature:
            for h in cls:
                h.errors = [i for i in range(len(x)) if h.classify(x[i]) == -y[i]]

    print('weak learners computing time:', time() - start)

    w = [1 / len(x)] * len(x)
    h = []

    for _ in range(T):
        wl = findWeakLearner(wls,w)
        print(wl)
        alpha = 0.5 * log((1-wl.error)/wl.error)
        h.append((wl, alpha))
        s = 0
        for i in range(len(x)):
            w[i] *= exp(-alpha * y[i] * wl.classify(x[i]))
            s += w[i]
        w = [wi / s for wi in w]

    print('adaboost time:', time() - start)
    return StrongLearner(h)

def drawBoundary(img, clf):
    dst = img.copy()
    rows, cols = img.shape[0], img.shape[1]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(rows):
        for j in range(cols):
            dst[i,j] = [255,0,255] if clf.classify((i,j, hsv[i,j,0], hsv[i,j,1], hsv[i,j,2])) == 1 else [0,0,255]
    return dst

trainSet = 'data_road/training/image_2'
groundTruth = 'data_road/training/gt_image_2'
testSet = 'data_road/testing/image_2'

sl = adaboost(trainSet, groundTruth, T=30)
print('error: ', sl.error())

img = cv2.imread('data_road/testing/image_2/um_000000.png', 1)
res = drawBoundary(img, sl)
cv2.imshow('result1', res)
cv2.imshow('hough', cv2.HoughLinesP(res, rho=2, theta=pi/120, threshold=120, lines=np.array([]), minLineLength=20, maxLineGap=35))

img = cv2.imread('data_road/testing/image_2/um_000040.png', 1)
cv2.imshow('result2', drawBoundary(img, sl))
cv2.waitKey()