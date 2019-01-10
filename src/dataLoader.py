import os
import keras
import numpy as np
import pandas as pd

from scipy import ndimage
from skimage import io, transform


class DataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, csv_file, root_dir, batch_size, transform=None, shuffle=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hc_frame = pd.read_csv(csv_file)
        self.list_IDs = list_IDs
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.hc_frame))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_idx):
        batch_x = self.hc_frame.iloc[list_idx, 0]
        # Output annotation or distance transformation
        if 'distanceTransform' in self.transform.keys() and self.transform['distanceTransform']:
            batch_y = self.hc_frame.iloc[list_idx, 0].str.replace('.png', '_DistanceTransform.png')
            output_mode = 'distanceTransform'
        else:
            batch_y = self.hc_frame.iloc[list_idx, 0].str.replace('.png', '_Annotation.png')
            output_mode = 'annotation'

        # calculated head circumference in pixel =  head circumference (mm) // pixel_size(mm)
        # normalize hc to [-1,1] choosing 150 as max
        batch_hc = (self.hc_frame.iloc[list_idx, 8] / self.hc_frame.iloc[list_idx, 1] - (
        75 / self.hc_frame.iloc[list_idx, 1])) / (75 / self.hc_frame.iloc[list_idx, 1])

        # center_x_pixel = center_x_mm / pixel_mm
        batch_center_x = self.hc_frame.iloc[list_idx, 3] / self.hc_frame.iloc[list_idx, 1]
        # center_y_pixel = center_y_mm / pixel_mm
        batch_center_y = self.hc_frame.iloc[list_idx, 4] / self.hc_frame.iloc[list_idx, 1]

        # semi_a_pixel = semi_a_mm/ pixel_mm
        batch_a = self.hc_frame.iloc[list_idx, 5] / self.hc_frame.iloc[list_idx, 1]
        # semi_b_pixel = semi_b_mm/ pixel_mm
        batch_b = self.hc_frame.iloc[list_idx, 6] / self.hc_frame.iloc[list_idx, 1]

        # angle_rad
        batch_angle = self.hc_frame.iloc[list_idx, 7]

        # read input image x
        X = np.array([self.image_transformer(
            np.expand_dims(io.imread(os.path.join(self.root_dir, file_name)), axis=3), 'image') for file_name in
            batch_x])
        # read output image y
        Y = np.array([self.image_transformer(
            np.expand_dims(io.imread(os.path.join(self.root_dir, file_name)), axis=3), output_mode) for file_name in
            batch_y])

        # normalize the vars to [-1,1]
        normelize_center_x = int(self.transform['reshape'][0]/2) if 'reshape' in self.transform.keys() else 400
        normelize_center_y = int(self.transform['reshape'][1]/2) if 'reshape' in self.transform.keys() else 270
        normelize_semi_a = normelize_center_x
        normelize_semi_b = normelize_center_x

        # Ellipse parameter to numpy array
        CX = np.transpose(np.array([(batch_center_x - normelize_center_x) / normelize_center_x]))
        CY = np.transpose(np.array([(batch_center_y - normelize_center_y) / normelize_center_y]))
        A = np.transpose(np.array([(batch_a - normelize_semi_a) / normelize_semi_a]))
        B = np.transpose(np.array([(batch_b - normelize_semi_b) / normelize_semi_b]))
        SIN = np.transpose(np.sin(np.array([batch_angle])))
        COS = np.transpose(np.cos(np.array([batch_angle])))
        HC = np.transpose(np.array([batch_hc]))

        return X, [Y, CX, CY, A, B, SIN, COS, HC]

    # transformer for image augmentation
    def image_transformer(self, image, mode):
        # reshape image
        if 'reshape' in self.transform.keys():
            image = self.reshape(image, self.transform['reshape'])
        # fill ellipse
        if mode == 'annotation':
            image = ndimage.binary_fill_holes(image[:, :]).astype(int)

        return image

    def reshape(self, img, output_size):
        assert isinstance(output_size, (int, tuple))

        new_h, new_w = output_size[0], output_size[1]
        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(img, (new_h, new_w))
