import os
import keras
import numpy as np
import pandas as pd

from skimage import io, transform, color


class DataGenerator(keras.utils.Sequence):
    def __init__(self, csv_file, root_dir, batch_size, transform=None, shuffle=False):
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
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.hc_frame) / self.batch_size))

    def __getitem__(self, index):
          'Generate one batch of data'
          # Generate indexes of the batch
          indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

          # Generate data
          X, Y = self.__data_generation(indexes)

          return X,Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.hc_frame))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_idx):
        batch_x = self.hc_frame.iloc[list_idx,0]
        batch_y = self.hc_frame.iloc[list_idx,0].str.replace('.png', '_Annotation.png')

        # head circumference in pixel = head circumference (mm) // pixel_size(mm)
        batch_hc = self.hc_frame.iloc[list_idx,2] // self.hc_frame.iloc[list_idx,1]

        # read input image x
        X = np.array([self.image_transformer(
            color.gray2rgb(io.imread(os.path.join(self.root_dir, file_name)))) for file_name in batch_x])
        # read output image y
        Y = np.array([self.image_transformer(
            color.gray2rgb(io.imread(os.path.join(self.root_dir, file_name)))) for file_name in batch_y])

        # batch_hc to numpy array
        HC = np.array(batch_hc)

        return X, [Y, HC]

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


