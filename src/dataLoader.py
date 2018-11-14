import numpy as np
import keras
import linecache
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 path='../Dataset/trainingtraining_set_pixel_size_and_HC.csv', shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.path = path
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = list()
        Y = list()

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # read line from csv
            data = linecache.getline('foo.csv', i).split(',')

            # append output (input image)
            img_x = cv2.imread(self.path + 'set/' + data[i,0])
            X.append({'img_x': img_x})

            # append output (segmantated image), (head circumference (mm)), (pixel size(mm))
            img_y =  cv2.imread(self.path + 'set/' + data[i,0].replace('.png', '') + '_HC_Annotation.png')
            Y.append({'img_y': img_y, 'hc': data[i,2], 'pixel': data[i,1]})

        return X, Y
