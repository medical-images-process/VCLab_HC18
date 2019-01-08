import os
import cv2
import csv
import numpy as np

from math import pi, radians, sqrt
from scipy import ndimage

def ellipseParameter(path) :
    # Load image and plot
    img = cv2.imread(path,0)

    # get ellipse parameter
    x, y, a, b, theta = 0,0,0,0,0
    ret,thresh = cv2.threshold(img,127,255,0)
    methods = [cv2.CONTOURS_MATCH_I1, cv2.CONTOURS_MATCH_I2]
    for m in methods:
        im2,contours,hierarchy = cv2.findContours(thresh, m , cv2.INTERSECT_FULL)
        cnt = contours[0]
        (xx,yy),(MA,ma),angle = cv2.fitEllipse(cnt)
        if MA > ma:
            aa, bb = MA, ma
            angle = angle
        else:
            aa, bb = ma, MA
            angle =  (angle - 90) % 360
        a += aa/2
        b += bb/2
        x += xx
        y += yy
        theta += angle
    l = len(methods)
    return (x/l, y/l), (a/l, b/l), theta/l


def circumference(a, b):
    l = (a-b) / (a+b)
    return (a+b) * pi * (1 + ((3 * l**2 ) / (10 + sqrt(4-3*l**2))))

# image path
path = os.path.join(os.getcwd(),'Dataset', 'training')


def ellipseDistanceTransformation(path):
    # Load image and plot
    img = cv2.imread(path, 0)
    img = cv2.bitwise_not(img)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
    return dist

with open(os.path.join(path, 'training_set_pixel_size_and_HC.csv'), "rt") as infile, open(os.path.join(path, 'training.csv'), "w", newline='') as outfile:
   reader = csv.reader(infile)
   writer = csv.writer(outfile)
   # write header
   headers = next(reader, None)
   if headers:
       writer.writerow(headers + ['center_x_mm'] + ['center_y_mm'] + ['semi_axes_a_mm'] + ['semi_axes_b_mm'] + ['angle_rad'] + ['hc_calc_mm'])
   for row in reader:
       img_name = row[0].replace('.png', '_Annotation.png')
       pixel_mm = float(row[1])
       (x, y), (a, b), theta = ellipseParameter(os.path.join(path, 'set', img_name))
       img_distance = ellipseDistanceTransformation(os.path.join(path, 'set', img_name))
       cv2.imwrite(os.path.join(path, 'set', img_name.replace('Annotation', 'DistanceTransform')), img_distance)
       writer.writerow(row +
                       [str(round(x*pixel_mm,9))] +
                       [str(round(y*pixel_mm,9))] +
                       [str(round(a*pixel_mm,9))] +
                       [str(round(b*pixel_mm,9))] +
                       [str(round(radians(theta),9))] +
                       [str(round(circumference(a,b)*pixel_mm, 2))])


import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
# get ellipse parameters
pixel_mm = 0.190535875467

img_path = os.path.join(path, 'set', '388_HC_Annotation.png')
(x,y), (a, b), theta = ellipseParameter(img_path)
print('x,y: ' + str(x*pixel_mm) + ',' + str(y*pixel_mm) + ' a: '+str(a*pixel_mm)+' b: '+str(b*pixel_mm)+' theta: ' + str(theta) + ' rad: ' + str(radians(theta)))
print('hc: ' + str(round(circumference(a,b)*pixel_mm, 2)))
angles = [theta ]
im = cv2.imread(img_path, 0)
im2 = np.ndarray(im.shape)
ells = Ellipse((x,y), a*2,b*2 , theta,  edgecolor='red', facecolor='none',)
fig = plt.figure()
fig.add_subplot(111)

# plt.imshow(im, cmap='gray')
# ax = plt.gca()
# ax.add_patch(ells)
# plt.show()

#img_fill_holes = ndimage.binary_fill_holes(im[:,:]).astype(int)
#plt.imshow(img_fill_holes, cmap='gray')
#plt.show()

dist = ellipseDistanceTransformation(img_path)
print(dist)
plt.imshow(dist, cmap='gray')
plt.show()