import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import *
from predictor import predict
from models.u_net import get_unet
from keras.callbacks import ModelCheckpoint, TensorBoard


def main(argv):
    ####################################################################################################################
    # Parameters                                                                                                       #
    ####################################################################################################################
    learn_mode = {
        'load': True,
        'train': True,
        'evaluate': False,
        'predict': True
    }
    data_mode = 'inv'
    input_shape = (256, 384, 1)
    pooling_mode = 'avg'

    ###############################
    # training parameter
    num_epochs = 30
    batch_size = 2
    steps_per_epoch = 1000
    lr = 1e-4
    verbose = 1

    # normelize elipse parameters to [-1, 1]

    normelizer = {
        'hc': 200,
        'cx': 400,
        'cy': 270,
        'sa': 400,
        'sb': 400,
        "scale": 2.1
    }
    ###############################
    #  paths
    # model paths
    model_name = 'unet_' + str(input_shape[0:2]).replace(' ', '').replace('(', '').replace(')', '').replace(',',
                                                                                                            'x')
    # model_name = 'trained_model_mnist'
    model_path = os.path.join(os.getcwd(),
                              'saved_models/' + model_name + '.h5')
    # train set path
    train_dataset_dir = os.path.join(os.getcwd(), "Dataset", "training")
    test_dataset_dir = os.path.join(os.getcwd(), "Dataset", "test")
    traindf = prepareParameter(pd.read_csv(os.path.join(train_dataset_dir, 'training.csv')), norm=normelizer)

    ###############################
    #  dataset parameters
    len_set = int(len(traindf))

    ####################################################################################################################
    # Load Model                                                                                                       #
    ####################################################################################################################
    # load model
    print("Load model")
    model_template, model = get_unet(model_name=model_name, pooling_mode=pooling_mode, input_shape=input_shape, lr=lr)
    # details of the model
    # model_template.summary()

    if learn_mode['load']:
        print('Load saved parameters: ' + model_path)
        model_template.load_weights(model_path)

    ####################################################################################################################
    # Train Model                                                                                                      #
    ####################################################################################################################
    if learn_mode['train']:
        ################################################################################################################
        # Load Dataset                #
        print("Load dataset")
        # data_gen_args = dict(rotation_range=0.2,
        #                     width_shift_range=0.05,
        #                     height_shift_range=0.05,
        #                     shear_range=0.05,
        #                     zoom_range=0.05,
        #                     horizontal_flip=True,
        #                     fill_mode='nearest')
        # training_generator = preTrainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             fill_mode='nearest')

        training_generator = trainGenerator(batch_size, 'Dataset/training/', 'image', data_mode, traindf,
                                            data_gen_args, save_to_dir=None)

        ################################################################################################################
        # Start Training              #

        # set checkpoints
        file = "saved_models/checkpoints/ckp-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath=file, monitor='loss', verbose=verbose, mode='auto',
                                     save_best_only=True)

        # set up tensorboard
        tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        callback_list = [checkpoint, tensorboard]

        ################################################################################################################
        # train model
        print("Train model")
        model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

        model.fit_generator(training_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                            callbacks=[model_checkpoint])

        # Save model via template_model (shares the same weights):
        model_template.save(model_path)
        print("Model saved to disk: " + model_path)

    ####################################################################################################################
    # Evaluate Model                                                                                                   #
    ####################################################################################################################
    if learn_mode['evaluate']:
        print("Evaluate model")

    ####################################################################################################################
    # Predict Test                                                                                                     #
    ####################################################################################################################
    if learn_mode['predict']:
        print('Predict...')
        # testGene = preTestGenerator("data/membrane/test")
        # results = model.predict_generator(testGene, 30, verbose=1)
        # preSaveResult("data/membrane/test", results[0])
        csv_file = os.path.join(test_dataset_dir, 'test_set_pixel_size.csv')
        # csv_file = os.path.join(train_dataset_dir, 'training.csv')

        predict(model=model, path=test_dataset_dir,
                csv_in=csv_file, norm=normelizer)

        ################################################################################################################
        # Delete the  Model                                                                                            #
        ################################################################################################################
        print("Delete model")
        del model, model_template
        print('***** FINISHED *****')


def prepareParameter(df, norm):
    df['annotation'] = df['filename'].replace('_HC.png', '_HC_annotation.png')
    df['distanceTransform'] = df['filename'].replace('_HC.png', '_HC_distanceTransform.png')
    df['hc'] = ((df['head circumference (mm)'] / df['pixel size(mm)']  / norm["scale"])- norm['hc']) / norm['hc']
    df['cx'] = ((df['center_x_mm'] / df['pixel size(mm)'] / norm["scale"]) - norm['cx']) / norm['cx']
    df['cy'] = ((df['center_y_mm'] / df['pixel size(mm)'] / norm["scale"]) - norm['cy']) / norm['cy']
    df['sa'] = ((df['semi_axes_a_mm'] / df['pixel size(mm)'] / norm["scale"]) - norm['sa']) / norm['sa']
    df['sb'] = ((df['semi_axes_b_mm'] / df['pixel size(mm)'] / norm["scale"]) - norm['sb']) / norm['sb']
    df['sin'] = np.sin(df['angle_rad'])
    df['cos'] = np.cos(df['angle_rad'])
    df.drop(['head circumference (mm)', 'center_x_mm', 'center_y_mm', 'semi_axes_a_mm', 'semi_axes_b_mm', 'angle_rad'],
            axis=1)
    return df


if __name__ == "__main__":
    main(sys.argv)
