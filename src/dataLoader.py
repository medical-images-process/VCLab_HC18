from skimage import io, transform, color
import os
import keras
import pandas as pd
import numpy as np
class DataGenerator(keras.utils.Sequence):
    def __init__(self, csv_file, root_dir, batch_size, transform=None, output_size=512):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hc_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transform


    def __len__(self):
        return len(self.hc_frame)

    def __getitem__(self, idx):
        batch_x = self.hc_frame.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, 0]
        batch_y = self.hc_frame.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, 0].str.replace('.png', '_Annotation.png')

        # head circumference in pixel = head circumference (mm) // pixel_size(mm)
        batch_hc = self.hc_frame.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, 2] // \
                   self.hc_frame.iloc[idx * self.batch_size:(idx + 1) * self.batch_size, 1]

        # read input image x
        X = np.array([self.image_transformer(
            color.gray2rgb(io.imread(os.path.join(self.root_dir, file_name)))) for file_name in batch_x])
        # read output image y
        Y = np.array([self.image_transformer(
            color.gray2rgb(io.imread(os.path.join(self.root_dir, file_name)))) for file_name in batch_y])

        # batch_hc to numpy array
        HC = np.array(batch_hc)

        # X = self.image_transformer(io.imread(os.path.join(self.root_dir, self.hc_frame.iloc[idx, 0])))
        # Y = self.image_transformer(io.imread(os.path.join(self.root_dir, self.hc_frame.iloc[idx, 0].replace('.png', '_Annotation.png'))))
        return X, Y

    # transformer for image augmentation
    def image_transformer(self, image):
        # reshape image
        if 'reshape' in self.transform.keys():
            image = self.reshape(image, self.transform['reshape'])
        return image


    def reshape(self, img, output_size):
        assert isinstance(output_size, (int, tuple))
        h, w = img.shape[:2]
        if isinstance(output_size, int):
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size,output_size

        new_h, new_w = output_size, output_size
        new_h, new_w = int(new_h), int(new_w)

        return transform.resize(img, (new_h, new_w))


