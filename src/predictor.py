import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform

def predict(model, csv, im_path):
    hc_frame = pd.read_csv(csv)
    #for i in len(hc_frame):
    for i in range(2):
        p = io.imread(os.path.join(im_path, hc_frame.iloc[i,0]))
        p = transform.resize(p, (512,512))
        p = np.expand_dims(np.expand_dims(p, axis=3), axis=0)
        annotation, cx, cy, a, b, angle_sin, angle_cos, hc =model.predict_on_batch(p)
        angle_rad = np.arctan2(angle_sin, angle_cos)

        plt.imshow(annotation[0,:,:,0], cmap='gray')
        plt.show()
