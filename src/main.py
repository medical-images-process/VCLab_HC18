import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from src.predictor import predict
from src.dataLoader import DataGenerator
from src.models.u_net import get_unet


def main(argv):
    ###############################
    # Parameters                  #
    ###############################
    # leran mode = train | return_from_checkpoint | evaluate_only | predict_only
    learn_mode = 'train'
    input_shape = (540, 800, 1)
    pooling_mode = 'avg'
    image_transformer = {'reshape': input_shape[0:2], 'distanceTransform': False}
    model_name = 'unet_min_' + str(input_shape[0:2]).replace(' ', '').replace('(', '').replace(')', '').replace(',',
                                                                                                                'x')
    ###############################
    #  paths
    train_dataset_dir = os.path.join(os.getcwd(), "Dataset", "training")
    csv_file = os.path.join(train_dataset_dir, 'training.csv')

    test_dataset_dir = os.path.join(os.getcwd(), 'Dataset/test')
    test_csv = os.path.join(train_dataset_dir, 'test_set_pixel_size.csv')

    model_path = os.path.join(os.getcwd(), 'saved_models/keras_'+ model_name + 'trained_model.h5')
    return_checkpoint = os.path.join(os.getcwd(), 'saved_models/checkpoints/ckp-02-4.93.hdf5')

    ###############################
    # training parameter
    num_epochs = 30
    batch_size = 8
    num_workers = 4
    shuffle = True
    trainings_split = 0.8
    len_set = int(len(pd.read_csv(csv_file)))
    num_training_samples = int(len_set * trainings_split)
    num_validation_samples = len_set - num_training_samples

    verbose = 1

    ###############################
    # Load Model                  #
    ###############################
    # load model
    print("Load model")
    model_template, model = get_unet(model_name=model_name, pooling_mode=pooling_mode, input_shape=input_shape)

    if learn_mode == 'return_from_checkpoint':
        print('Return from checkpoint: ' + return_checkpoint)
        model.load_weights(return_checkpoint)
        # continue training
        learn_mode = 'train'

    if learn_mode == 'predict_only':
        print('Predictions on model: ' + model_path)
        model.load_weights(model_path)

    # details of the model
    # from keras.utils import plot_model
    # plot_model(model_template, to_file='model.png')
    # model_template.summary()

    if learn_mode == 'train':
        ###############################
        # Load Dataset                #
        ###############################
        print("Load dataset")
        partition = {
            'train': np.arange(num_training_samples),
            'validate': (len_set - 1) - np.arange(num_validation_samples)}

        training_generator = DataGenerator(partition['train'], csv_file=csv_file,
                                           root_dir=os.path.join(train_dataset_dir, 'set'),
                                           batch_size=batch_size, transform=image_transformer)

        validation_generator = DataGenerator(partition['validate'], csv_file=csv_file,
                                             root_dir=os.path.join(train_dataset_dir, 'set'),
                                             batch_size=batch_size, transform=image_transformer)

        ###############################
        # Train Model                 #
        ###############################
        # set checkpoints
        file = "saved_models/checkpoints/ckp-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath=file, monitor='val_loss', verbose=verbose, mode='auto',
                                     save_best_only=True)
        callback_list = [checkpoint]

        # train model
        print("Train model")
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            steps_per_epoch=num_training_samples / batch_size,
                            epochs=num_epochs,
                            callbacks=callback_list,
                            verbose=verbose,
                            shuffle=shuffle)

        # Save model via template_model (shares the same weights):
        model_template.save(model_path)
        print("Model saved to disk")

        # evaluate training
        learn_mode = 'evaluate'


    if learn_mode == 'evaluate_only' or learn_mode == 'evaluate':
        ###############################
        # Evaluate Model              #
        ###############################
        print("Evaluate model")
        scores = model.evaluate_generator(validation_generator, num_validation_samples / batch_size,
                                          workers=num_workers)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        # predict model
        if learn_mode == 'evaluate':
            learn_mode  = 'predict'

    if learn_mode == 'predict' or learn_mode == 'predict_only':
        ###############################
        # Predict Test                #
        ###############################
        print('Predict...')
        predict(model=model, path=train_dataset_dir, csv_in=csv_file, image_transformer=image_transformer)

    ###############################
    # Delete the  Model           #
    ###############################
    print("Delete model")
    del model, model_template
    print('***** FINISHED *****')

if __name__ == "__main__":
    main(sys.argv)
