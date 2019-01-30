import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform


path = os.path.join(os.getcwd(), 'Dataset', 'training', 'set')

img_list = ['000_HC.png',
            '002_HC_Annotation.png',
            '022_HC_Annotation.png'
            ]
n = len(img_list)
cut = [14,523,16,784]
plt.figure(figsize=(64, 64))
for i, name in enumerate(img_list):
    img = io.imread(os.path.join(path, name))
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # img2 = img[28:,32:]
    # img2 = img[:512, 0:768
    img2 = img[14:523, 16:784]

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(img2, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

