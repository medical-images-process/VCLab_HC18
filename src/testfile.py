import os
import csv
import cv2
import numpy as np
from skimage import io, transform

path = os.path.join(os.getcwd(), "Dataset", "test")
csv_in = os.path.join(path, 'result.csv')

with open(csv_in, "rt") as infile:
    reader = csv.reader(infile)
    # write header
    headers = next(reader, None)
    for row in reader:
        # print('...' + row[0])
        pixel_mm = float(row[1])
        img_name = row[0]
        p = io.imread(os.path.join(os.path.join(path, 'image'), row[0]))


        cx = float(row[2]) / pixel_mm
        cy = float(row[3]) / pixel_mm
        sa = float(row[4]) / pixel_mm
        sb = float(row[5]) / pixel_mm
        an = float(row[6])
        hc = float(row[7]) / pixel_mm

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import Circle
        from matplotlib.patches import Ellipse

        plt.imshow(p, cmap='gray')
        ax = plt.gca()
        ells = Ellipse((cx, cy), sa, sb, an, edgecolor='red', facecolor='none', )
        mid = Circle((cx, cy), radius=5, color='red')
        ax.add_patch(ells)
        ax.add_patch(mid)
        plt.savefig(os.path.join(path, 'out', img_name))
        plt.cla()
