import sys
import numpy as np
import os
from numpy import *
from os import listdir
from PIL import Image
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt

sys.path.append('..')
data_dir = '/Users/zhangmingjie/Documents/Github/dcgan_tensorflow/data/mnist/data/'


def mnist():
    trY = []
    teY = []
    countx = 0
    county = 0
    folders = os.listdir(data_dir)
    print "========= loading images ========"
    for folder in folders:
        if folder == '.DS_Store':
            continue
        print 'loading images of label' + folder + '...'
        files = os.listdir(data_dir + folder)
        m = len(files)
        m = int(m / 3.0 * 2)
        for file in files[:m]:
            if file == '.DS_Store':
                continue
            image = Image.open(data_dir + folder + '/' + file)
            Im = array(image).reshape((1, 28 * 28))
            Im = 255 - Im
            if countx == 0:
                trX = Im
                countx += 1
            else:
                trX = np.row_stack((trX, Im))
            trY.append(int(folder))
        for file in files[m:]:
            image = Image.open(data_dir + folder + '/' + file)
            Im = array(image).reshape((1, 28 * 28))
            Im = 255 - Im
            if county == 0:
                teX = Im
                county += 1
            else:
                teX = np.row_stack((teX, Im))
            teY.append(int(folder))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    return trX, teX, trY, teY


def mnist_with_valid_set():
    trX, teX, trY, teY = mnist()

    train_inds = range(len(trX))
    np.random.shuffle(train_inds)
    trX = trX[train_inds]
    trY = trY[train_inds]
    # trX, trY = shuffle(trX, trY)
    vaX = teX[:]
    vaY = teY[:]
    trX = trX[:]
    trY = trY[:]

    return trX, vaX, teX, trY, vaY, teY
