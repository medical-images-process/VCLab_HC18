import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.predictor import predict
from src.models.u_net import get_unet
from src.dataLoader import DataGenerator
from keras.callbacks import ModelCheckpoint


def main(argv):
    ###############################
    # Parameters                  #
    ###############################
    # leran mode = train | return_from_checkpoint | evaluate_only | predict_only
    learn_mode = 'train'
    input_shape = (540, 800, 1)
    pooling_mode = 'avg'
    image_transformer = {'reshape': input_shape[0:2], 'distanceTransform': True}
    model_name = 'unet_min_' + str(input_shape[0:2]).replace(' ', '').replace('(', '').replace(')', '').replace(',',
                                                                                                                'x')

    ###############################
    # training parameter
    num_epochs = 500
    batch_size = 8
    shuffle = True
    lr = 0.0001
    verbose = 1

    ###############################
    #  paths
    train_dataset_dir = os.path.join(os.getcwd(), "Dataset", "training")
    csv_file = os.path.join(train_dataset_dir, 'training.csv')

    test_dataset_dir = os.path.join(os.getcwd(), 'Dataset/test')
    test_csv = os.path.join(train_dataset_dir, 'test_set_pixel_size.csv')

    model_path = os.path.join(os.getcwd(),
                              'saved_models/' + 'trained_model_' + model_name + '_' + str(num_epochs) + '_' + str(
                                  lr) + '.h5')
    return_checkpoint = os.path.join(os.getcwd(), 'saved_models/checkpoints/ckp-48-1.00.hdf5')

    ###############################
    #  dataset parameters
    trainings_split = 0.9
    len_set = int(len(pd.read_csv(csv_file)))
    num_training_samples = int(len_set * trainings_split)
    num_validation_samples = len_set - num_training_samples

    ###############################
    # Load Model                  #
    ###############################
    # load model
    print("Load model")
    model_template, model = get_unet(model_name=model_name, pooling_mode=pooling_mode, input_shape=input_shape, lr=lr)

    # details of the model
    # from keras.utils import plot_model
    # plot_model(model_template, to_file='model.png')
    # model_template.summary()

    if learn_mode == 'return_from_checkpoint':
        print('Return from checkpoint: ' + return_checkpoint)
        model.load_weights(return_checkpoint)
        # continue training
        learn_mode = 'train'

    if learn_mode == 'predict_only' or learn_mode == "evaluate":
        print('Predictions on model: ' + model_path)
        model.load_weights(model_path)

    if learn_mode == 'train' or learn_mode == 'train_only':
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
        checkpoint = ModelCheckpoint(filepath=file, monitor='loss', verbose=verbose, mode='auto',
                                     save_best_only=True)
        callback_list = [checkpoint]

        # train model
        print("Train model")
        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      steps_per_epoch=num_training_samples / batch_size,
                                      epochs=num_epochs,
                                      callbacks=callback_list,
                                      verbose=verbose,
                                      shuffle=shuffle)

        # Save model via template_model (shares the same weights):
        model_template.save(model_path)
        print("Model saved to disk")

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join('saved_models', model_name + '_' + str(num_epochs) + '.png'))

        # evaluate training
        if learn_mode == 'train':
            learn_mode = 'evaluate'

    ###############################
    # Evaluate Model              #
    ###############################
    if learn_mode == 'evaluate_only' or learn_mode == 'evaluate':
        print("Evaluate model")
        scores = model.evaluate_generator(validation_generator, num_validation_samples / batch_size)
        print("%s: %.2f%%" % (model.metrics_names[0], scores[0]))
        print("ImageOut: %.2f%%" % (scores[1]))
        print("Center_X: %.2f%%" % (scores[2]))
        print("Center_Y: %.2f%%" % (scores[3]))
        print("Semi_A: %.2f%%" % (scores[4]))
        print("Semi_B: %.2f%%" % (scores[5]))
        print("Sin: %.2f%%" % (scores[6]))
        print("Cos: %.2f%%" % (scores[7]))
        print("HC: %.2f%%" % (scores[8]))

        # predict model
        if learn_mode == 'evaluate':
            learn_mode = 'predict'

    if False and (learn_mode == 'predict' or learn_mode == 'predict_only'):
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
