import os
import csv
import cv2
import numpy as np
from skimage import io, transform


def predict(model, path, csv_in, image_transformer):
    # variable for undo normalisation to [-1,1]
    rn_x = image_transformer['reshape'][0] / 2 if 'reshape' in image_transformer.keys() else 400
    rn_y = image_transformer['reshape'][1] / 2 if 'reshape' in image_transformer.keys() else 270
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
            print('...' + row[0])
            pixel_mm = float(row[1])
            img_name = row[0].replace('.png', '_Predicted.png')
            p = io.imread(os.path.join(os.path.join(path, 'set'), row[0]))
            p = transform.resize(p, image_transformer['reshape'][0:2])
            p = np.expand_dims(np.expand_dims(p, axis=3), axis=0)
            pimg, cx, cy, a, b, angle_sin, angle_cos, hc = model.predict_on_batch(p)
            cv2.imwrite(os.path.join(path, 'out', img_name), pimg[0, :, :, 0])
            cx = cx * rn_x + rn_x
            cy = cy * rn_y + rn_y
            a = a * rn_x + rn_x
            b = b * rn_x + rn_x
            angle_rad = np.arctan2(angle_sin, angle_cos)
            writer.writerow([row[0]] +
                            [str(round(cx[0][0] * pixel_mm, 9))] +
                            [str(round(cy[0][0] * pixel_mm, 9))] +
                            [str(round(a[0][0]  *pixel_mm, 9))] +
                            [str(round(b[0][0] * pixel_mm, 9))] +
                            [str(round(angle_rad[0][0], 9))] +
                            [str(round(hc[0][0] * pixel_mm, 2))])
