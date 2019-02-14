import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform



def sp_noise(image,map,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    output[map] = image[map]
    return output

path = os.path.join(os.getcwd(), 'Dataset', 'training', 'set')

img_list = ['099_HC.png',
            '099_HC_Annotation.png',
            '022_HC_Annotation.png'
            ]

img_hc = io.imread(os.path.join(path, img_list[0]))

# get relevant area
annotation = io.imread(os.path.join(path, img_list[1]))
inv = cv2.bitwise_not(annotation)
dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
map = dist < 25
map2 = dist == 0
# add noise to image
img = sp_noise(img_hc, map, 0.5)
img[map2] = annotation[map2]
img = transform.resize(img, (28,28)) * 255

cv2.imwrite(os.path.join(os.getcwd(), 'Dataset', 'test.png'), img)
cv2.imshow("image", img)
cv2.waitKey(0)


import pandas as pd

