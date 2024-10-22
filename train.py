from math import exp, inf, log
import os
import random
from time import time
import cv2
import multiprocessing
import numpy as np

from commons import WeakLearner, hist, is_inside, local_binary_patterns, imagesPath, groundTruthPath

def is_road(pixel):
    return not np.array_equal(pixel, [0,0,255])

def is_same_class(block):
    same = is_road(block[0,0])
    for line in block:
        for pixel in line:
            if is_road(pixel) is not same:
                return False
    return True

def feature_vector(img, lbp, i, j, vroad):
    vclass = hist(lbp[i-5:i+5, j-5:j+5])

    #compute average color
    avg_color = [0, 0, 0]
    for x in range(i-5, i+5):
        for y in range(j-5, j+5):
            b, g, r = img[x,y]
            avg_color[0] += b
            avg_color[1] += g
            avg_color[2] += r

    avg_hsv = [0, 0, 0]
    for line in cv2.cvtColor(img[i-5:i+5, j-5:j+5], cv2.COLOR_BGR2HSV):
        for pixel in line:
            b, g, r = pixel
            avg_hsv[0] += b
            avg_hsv[1] += g
            avg_hsv[2] += r
    
    vfinal = [i, j] + [channel // 100 for channel in avg_color + avg_hsv] + vclass
    d = [(0, 20), (20, 20), (20, 0), (20, -20), (0, -20), (-20, -20), (-20, 0), (-20, 20)]

    for dx, dy in d:
        nx, ny = i+dx, j+dy
        #concatenate with vcontext[i]
        vfinal += hist(lbp[nx-10:nx+10, ny-10:ny+10]) if is_inside(lbp, nx-10, ny-10) and is_inside(lbp, nx+10, ny+10) else [25] * 16

    #concatenate with vsupport, (vroad[1] - vclass) & (vroad[2] - vclass)
    return vfinal + hist(lbp[i-10:i+10, j-10:j+10]) + [abs(vr - vc) for vr, vc in zip(vroad[0], vclass)] + [abs(vr - vc) for vr, vc in zip(vroad[1], vclass)]

def image_feature_vectors(imageFile, groundTruthFile, k1, k2):
    print(imageFile)
    img = cv2.imread(imagesPath + '/' + imageFile, 1)[150:]
    gt = cv2.imread(groundTruthPath + '/' + groundTruthFile, 1)[150:]
    rows, cols = img.shape[0], img.shape[1]

    lbp = local_binary_patterns(img)
    #compute road blocks
    vroad = [hist(lbp[rows-20:rows, cols-50:cols-30]), hist(lbp[rows-20:rows, cols+30:cols+50])]

    #choose k road blocks and k non-road blocks
    set_road, set_nonroad = set(), set()

    while len(set_road) < k1 or len(set_nonroad) < k2:
        i = random.randrange(10, rows-9)
        j = random.randrange(10, cols-9)
        if is_same_class(gt[i-5:i+5, j-5:j+5]):
            if is_road(gt[i,j]):
                if len(set_road) < k1:
                    set_road.add((i,j))
            else:
                if len(set_nonroad) < k2:
                    set_nonroad.add((i,j))
    
    return [feature_vector(img, lbp, i, j, vroad) for i, j in list(set_road) + list(set_nonroad)]

def feature_vectors(k1, k2):
    imageFiles = os.listdir(imagesPath)
    groundTruthFiles = os.listdir(groundTruthPath)

    start = time()
    with multiprocessing.Pool() as pool, open('random_split.csv') as f:
        x = pool.starmap(image_feature_vectors, \
                              [(imageFiles[int(idx)], groundTruthFiles[int(idx)], k1, k2) for idx in f.readline().split(',')])
    print('compute feature vectors from all images time', time() - start)
    y = []
    for _ in x:
        y += [1] * k1 + [-1] * k2

    return [vect for sublist in x for vect in sublist], y

def find_weak_learner(x, y, w):
    best_h = WeakLearner(0,0,0,inf)

    for j in range(len(x[0])):
        h0 = WeakLearner(j, -1, -1, 0)
        h1 = WeakLearner(j, -1, 1, 0)

        correct0 = []
        for xi, yi, wi in zip(x,y,w):
            if yi == 1:
                correct0.append(True)
                h1.error += wi
            else:
                correct0.append(False)
                h0.error += wi

        d = {t: [] for t in set([xi[j] for xi in x]) }

        for i, xi in enumerate(x):
            d[xi[j]].append(i)
        
        for t, dt in d.items():
            h0.threshold = h1.threshold = t
            
            if h0.error < best_h.error:
                best_h = WeakLearner(h0.feature_i, h0.threshold, h0.class_label, h0.error)
            if h1.error < best_h.error:
                best_h = WeakLearner(h1.feature_i, h1.threshold, h1.class_label, h1.error)

            for i in dt:
                correct0[i] = not correct0[i]
                if correct0[i]:
                    h0.error -= w[i]
                    h1.error += w[i]
                else:
                    h0.error += w[i]
                    h1.error -= w[i]

    return best_h

def adaboost(x, y, T):
    start = time()
    w = [1 / len(x)] * len(x)

    with open('weak_learners.csv', 'w+') as f:
        for _ in range(T):
            wl = find_weak_learner(x,y,w)
            print(wl)
            alpha = 0.5 * log((1-wl.error)/wl.error)
            f.write(f'{wl.feature_i},{wl.threshold},{wl.class_label},{wl.error},{alpha}\n')
            s = 0
            for i in range(len(x)):
                w[i] *= exp(-alpha * y[i] * wl.classify(x[i]))
                s += w[i]
            w = [wi / s for wi in w]

    print('adaboost time:', time() - start)

if __name__ == '__main__':
    x, y = feature_vectors(100, 100) # 1.5 min
    adaboost(x, y, 25) # 3 min