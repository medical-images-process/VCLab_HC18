import os
import csv
import cv2
import numpy as np
from skimage import io, transform


def ellipseParameter(im):
    # get ellipse parameter
    img = cv2.resize(im, (540,800))
    x, y, a, b, theta = 0, 0, 0, 0, 0
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    methods = [cv2.CONTOURS_MATCH_I1, cv2.CONTOURS_MATCH_I2]
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    (xx, yy), (MA, ma), angle = cv2.fitEllipse(cnt)
    if MA > ma:
        aa, bb = MA, ma
        angle = angle
    else:
        aa, bb = ma, MA
        angle = (angle - 90) % 360
    a += aa / 2
    b += bb / 2
    x += xx
    y += yy
    theta += angle
    return (x, y), (a, b), theta


def predict(model, path, csv_in, norm):
    # variable for undo normalisation to [-1,1]
    with open(csv_in, "rt") as infile, open(
            os.path.join(path, 'results.csv'), "w", newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        # write header
        headers = next(reader, None)
        if headers:
            writer.writerow(['filename'] + ['center_x_mm'] + ['center_y_mm'] + ['semi_axes_a_mm'] + ['semi_axes_b_mm']
                            + ['angle_rad'] + ['hc_mm'])
        for row in reader:
            # print('...' + row[0])
            pixel_mm = float(row[1])
            img_name = row[0].replace('.png', '_Predicted_dist_tmp.png')
            p = io.imread(os.path.join(os.path.join(path, 'image'), row[0]))
            p = transform.resize(p, (256, 384))
            p = np.expand_dims(np.expand_dims(p, axis=3), axis=0)

            pimg, cx, cy, sa, sb, angle_sin, angle_cos, hc = model.predict_on_batch(p)
            pimg = transform.resize(pimg[0, :, :, 0] , (540, 800))
            cv2.imwrite(os.path.join(path, 'out', img_name), pimg * 255)

            cx = (cx * norm['cx'] + norm['cx'] * norm["scale"])
            cy = (cy * norm['cy'] + norm['cy'] * norm["scale"])
            sa = (sa * norm['sa'] + norm['sa'] * norm["scale"])
            sb = (sb * norm['sb'] + norm['sb'] * norm["scale"])
            hc = (hc * norm['hc'] + norm['hc'] * norm["scale"])
            # # (cx, cy), (sa, sb), angle = ellipseParameter(pimg*255)
            # import matplotlib.pyplot as plt
            # from matplotlib.pyplot import Circle
            # plt.imshow(pimg, cmap='gray')
            # ax = plt.gca()
            # ax.add_patch(Circle((cx, cy), radius=5, color='red'))
            # plt.savefig(os.path.join(path, 'out', img_name))
            # plt.cla()

            angle_rad = np.arctan2(angle_sin, angle_cos)
            writer.writerow([row[0]] +
                            [str(round(cx[0][0] * pixel_mm, 9))] +
                            [str(round(cy[0][0] * pixel_mm, 9))] +
                            [str(round(sa[0][0] * pixel_mm, 9))] +
                            [str(round(sb[0][0] * pixel_mm, 9))] +
                            [str(round(angle_rad[0][0], 9))] +
                            [str(round(hc[0][0] * pixel_mm, 2))])
