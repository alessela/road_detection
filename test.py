import os
from time import time

import cv2
from commons import StrongLearner, WeakLearner, hist, is_inside, local_binary_patterns, imagesPath, groundTruthPath
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

def read_strong_classifier(T):
    sl = StrongLearner([])
    with open('weak_learners.csv') as f:
        for line in f.readlines()[:T]:
            splits = line.split(',')
            sl.hs.append((WeakLearner(int(splits[0]), int(splits[1]), int(splits[2]), float(splits[3])), float(splits[4])))
    return sl

def feature_vector(img, lbp, h, i, j, vroad):
    vclass = hist(lbp[i-5:i+5, j-5:j+5])
    
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
        vfinal += h[nx // 10 - 1][ny // 10 - 1] if is_inside(lbp, nx-10, ny-10) and is_inside(lbp, nx+10, ny+10) else [25] * 16

    #concatenate with vsupport, (vroad[1] - vclass) & (vroad[2] - vclass)
    return vfinal + h[i // 10 - 1][j // 10 - 1] + [abs(vr - vc) for vr, vc in zip(vroad[0], vclass)] + [abs(vr - vc) for vr, vc in zip(vroad[1], vclass)]

def labels(img):
    rows, cols = img.shape[0], img.shape[1]
    labels = [[0] * cols for _ in img]
    label = 0
    edges = [[] for _ in range(1000)]

    d = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for i in range(rows):
        for j in range(cols):
            if img[i, j] and labels[i][j] == 0:
                l = []
                for di, dj in d:
                    ni, nj = i + di, j + dj
                    if is_inside(img, ni, nj) and img[i,j] and labels[ni][nj] > 0:
                        l.append(labels[ni][nj])
                if l:
                    x = min(l)
                    labels[i][j] = x
                    for y in l:
                        if y != x:
                            edges[x].append(y)
                            edges[y].append(x)
                else:
                    label += 1
                    labels[i][j] = label
    
    newlabel = 0
    newlabels = [0] * (label + 1)
    for i in range(1,label+1):
        if newlabels[i] == 0:
            newlabel += 1
            newlabels[i] = newlabel
            q = [i]
            while q:
                x = q.pop(0)
                for y in edges[x]:
                    if newlabels[y] == 0:
                        newlabels[y] = newlabel
                        q.append(y)
    
    return [[newlabels[label] for label in line] for line in labels]

def classify(img: cv2.Mat, clf: StrongLearner):
    rows, cols = img.shape[0], img.shape[1]

    lbp = local_binary_patterns(img)
    h = [[hist(lbp[i:i+20, j:j+20]) for j in range(0, cols-19, 10)] for i in range(0, rows-19, 10)]
    h_rows, h_cols = len(h), len(h[0])

    #compute road blocks
    mid = h_cols // 2
    vroad = [h[-1][mid-2], h[-1][mid+2]]
    road_blocks = np.zeros((h_rows, h_cols), np.bool_)

    for hi in range(h_rows):
        i = (hi + 1) * 10
        for hj in range(h_cols):
            j = (hj + 1) * 10
            if clf.classify(feature_vector(img, lbp, h, i, j, vroad)) == 1:
                road_blocks[hi, hj] = True
    
    #post-processing
    lbls = labels(road_blocks)

    for hi in range(h_rows):
        for hj in range(h_cols):
            if lbls[hi][hj] != lbls[-1][mid]:
                road_blocks[hi, hj] = False
    
    for hi in range(h_rows):
        prev_j = -6
        for hj in range(h_cols):
            if road_blocks[hi, hj]:
                if hj - prev_j < 6:
                    for hj1 in range(prev_j+1, hj):
                        road_blocks[hi, hj1] = True
                prev_j = hj
    
        if h_cols - prev_j < 6:
            for hj in range(prev_j + 1, h_cols):
                road_blocks[hi, hj] = True

    for hj in range(h_cols):
        prev_i = -6
        for hi in range(h_rows):
            if road_blocks[hi, hj]:
                if hi - prev_i < 6:
                    for hi1 in range(prev_i+1, hi):
                        road_blocks[hi1, hj] = True
                prev_i = hi
        
        if h_rows - prev_i < 6:
            for hi in range(prev_i + 1, h_rows):
                road_blocks[hi, hj] = True
                
    return road_blocks

def draw_boundary(imageFile, clf: StrongLearner):
    img = cv2.imread(imagesPath + '/' + imageFile, 1)
    road_blocks = classify(img[150:], clf)
    dst = img.copy()

    for hi in range(len(road_blocks)):
        i = (hi + 1) * 10
        for hj in range(len(road_blocks[0])):
            if road_blocks[hi, hj]:
                j = (hj + 1) * 10
                for x in range(i-5, i+5):
                    for y in range(j-5, j+5):
                        dst[x + 150,y] = [255, 0, 255]

    return dst

def image_scores(imageFile, gtFile, clf: StrongLearner):
    img = cv2.imread(imagesPath + '/' + imageFile, 1)[150:]
    gt = cv2.imread(groundTruthPath + '/' + gtFile, 1)[150:]
    road_blocks = classify(img, clf)

    tp = fn = fp = 0
    for hi in range(len(road_blocks)):
        i = (hi + 1) * 10
        for hj in range(len(road_blocks[0])):
            j = (hj + 1) * 10
            if road_blocks[hi, hj]:
                for x in range(i-5, i+5):
                    for y in range(j-5, j+5):
                        if np.array_equal(gt[x,y], [255,0,255]):
                            tp += 1
                        else:
                            fp += 1
            else:
                for x in range(i-5, i+5):
                    for y in range(j-5, j+5):
                        if np.array_equal(gt[x,y], [255,0,255]):
                            fn += 1

    if tp == 0:
        return 0, 0, 0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * recall * precision / (recall + precision)

    return f_measure, precision, recall

def scores(clf: StrongLearner, category=''):
    imageFiles = os.listdir(imagesPath)
    groundTruthFiles = os.listdir(groundTruthPath)

    with open('random_split.csv') as f:
        imagesSet = [(imageFiles[i], groundTruthFiles[i]) for i in list(map(int, f.readlines()[1].split(','))) \
                     if imageFiles[i].startswith(category)]
    
    with multiprocessing.Pool() as pool:
        f_ms, ps, rs = 0, 0, 0
        for f_m, p, r in pool.starmap(image_scores, [(file, gt_file, clf) for file, gt_file in imagesSet]):
            f_ms += f_m
            ps += p
            rs += r

        return f_ms / len(imagesSet), ps / len(imagesSet), rs / len(imagesSet) 

def maximum_scores(clf: StrongLearner, category=''):
    max_scores = (0, 0, 0)
    max_threshold = 0
    start = time()

    thresholds = [-0.5, -0.25, 0, 0.25, 0.5]
    fms = []
    ps = []
    rs = []

    for threshold in thresholds:
        clf.threshold = threshold
        f_m, p, r = scores(clf, category)
        fms.append(f_m)
        ps.append(p)
        rs.append(r)

        print(f'threshold {threshold} - f-measure: {f_m}, precision: {p}, recall: {r},')
        if max_scores[0] < f_m:
            max_scores = (f_m, p, r)
            max_threshold = threshold
    
    print('compute maxf time:', time() - start)
    clf.threshold = max_threshold

    # plt.xlabel('Threshold')
    # plt.plot(thresholds, fms, label = 'F_measure')
    # plt.plot(thresholds, ps, label = 'Precision')
    # plt.plot(thresholds, rs, label = 'Recall')
    # plt.legend()
    # plt.savefig('results/thresholds.png')

    return max_scores[0], max_scores[1], max_scores[2]

def average_exec_time(clf: StrongLearner, category=''):
    imageFiles = os.listdir(imagesPath)
    avg_t = 0

    with open('random_split.csv') as f:
        imagesSet = [imageFiles[i] for i in list(map(int, f.readlines()[1].split(','))) \
                     if imageFiles[i].startswith(category)]

    for imageFile in imagesSet:
        start = time()
        classify(cv2.imread(imagesPath + '/' + imageFile, 1)[150:], clf)
        exec_time = time() - start
        avg_t += exec_time
    
    return avg_t / len(imagesSet)

def track_threshold(image, clf):  
    for threshold in [-0.5, -0.25, 0, 0.25, 0.5]:
        clf.threshold = threshold
        cv2.imwrite(f'results/th{threshold}.png', draw_boundary(image, clf))

def track_number_of_wls(image):
    Ts = list(range(5,26,5))
    fms = []
    ps = []
    rs = []

    for T in range(5,26,5):
        clf = read_strong_classifier(T)
        mf, p, r = maximum_scores(clf)
        cv2.imwrite(f'results/no_wls_{T}.png', draw_boundary(image, clf))

        fms.append(mf)
        ps.append(p)
        rs.append(r)
    
    plt.xlabel('No. of weak learners')
    plt.plot(Ts, fms, label = 'F_measure')
    plt.plot(Ts, ps, label = 'Precision')
    plt.plot(Ts, rs, label = 'Recall')
    plt.legend()
    plt.savefig('results/no_wls.png')

if __name__ == '__main__':

    T=25
    clf = read_strong_classifier(T)
    clf.threshold = 0.25
    imageFiles = os.listdir(imagesPath)
    # cv2.imshow('test', draw_boundary(imageFiles[221], clf))

    category = 'uu_'
    # track_number_of_wls(imageFiles[221])
    print(average_exec_time(clf, category))

    # max_f, max_p, max_r = maximum_scores(clf, category) # ~ 2 min
    # print(f'f-measure: {max_f}, precision: {max_p}, recall: {max_r}, threshold: {clf.threshold}')
    # clf.threshold = 0.25
    # print(scores(clf))

    # cv2.imwrite('results/maxf1.png', draw_boundary(imageFiles[130], clf))
    # cv2.imwrite('results/maxf2.png', draw_boundary(imageFiles[25], clf))
    # cv2.imwrite('results/maxf3.png', draw_boundary(imageFiles[20], clf))
    
    # with open('random_split.csv') as f:
    #     random_split =  list(map(int, f.readlines()[1].split(',')))
  
    # imageFiles = [file for i, file in enumerate(os.listdir(imagesPath)) if i in random_split and file.startswith(category)]
    # cv2.imwrite(f'results/maxf_{category}1.png', draw_boundary(imageFiles[0], clf))
    # cv2.imwrite(f'results/maxf_{category}2.png', draw_boundary(imageFiles[1], clf))
    # cv2.imwrite(f'results/maxf_{category}3.png', draw_boundary(imageFiles[2], clf))
    cv2.waitKey(0)