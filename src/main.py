import os
import sys
import numpy as np

from src.predictor import predict
from keras.callbacks import ModelCheckpoint
from src.parallel_model_save import ModelCheckpointParallel
from src.dataLoader import DataGenerator
# from src.models.u_net import get_advanced_unet_512 as get_unet
from src.models.u_net import get_opt_advanced_unet_512 as get_unet


def main(argv):
    ###############################
    # Parameters                  #
    ###############################
    # paths
    train_dataset_dir = os.path.join(os.getcwd(),"Dataset", "training")
    csv_file = os.path.join(train_dataset_dir, 'training.csv')

    test_dataset_dir = os.path.join(os.getcwd(), 'Dataset/test')
    test_csv = os.path.join(train_dataset_dir, 'test_set_pixel_size.csv')

    model_path = os.path.join(os.getcwd(), 'saved_models/keras_unet_trained_model.h5')

    # training parameter
    num_epochs = 100
    batch_size = 2
    num_workers = 4
    shuffle = False
    trainings_split = 0.5
    #len_set = int(len(pd.read_csv(csv_file)))
    len_set = 16
    num_training_samples = int(len_set -len_set * trainings_split)
    num_validation_samples = int(len_set * trainings_split)

    verbose=1

    ###############################
    # Load Dataset                #
    ###############################
    print("load dataset")
    partition = {
        'train': np.arange(num_training_samples),
        'validate': len_set - np.arange(num_validation_samples)}
    imager_transformer = {'reshape': 512, 'distanceTransform': False}

    training_generator = DataGenerator(partition['train'], csv_file=csv_file, root_dir=os.path.join(train_dataset_dir, 'set'),
                                             batch_size= batch_size, transform=imager_transformer)

    validation_generator = DataGenerator(partition['validate'], csv_file=csv_file,
                                       root_dir=os.path.join(train_dataset_dir, 'set'),
                                       batch_size=batch_size, transform=imager_transformer)


    ###############################
    # Start learning              #
    ###############################

    # load model
    print("load model")
    model_template, model = get_unet()

    # details of the model
    model_template.summary()
    # from keras.utils import plot_model
    # plot_model(model_template, to_file='model.png')

    # checkpoint
    file = "saved_models/checkpoints/ckp-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath=file, monitor='val_loss', verbose=verbose, mode='auto', save_best_only=True)
    callback_list = [checkpoint]

    # train model
    print("train model")
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=num_training_samples / batch_size,
                        epochs=num_epochs,
                        callbacks=callback_list,
                        verbose=verbose,
                        shuffle=shuffle)

    # Save model via the template model (which shares the same weights):
    model_template.save(model_path)
    print("model saved to disk")

    # Predict
    predict(model=model, csv=csv_file, im_path=os.path.join(train_dataset_dir, 'set'))
    # evaluate the model
    print("evaluate model")
    scores = model.evaluate_generator(validation_generator, num_validation_samples/batch_size, workers=num_workers)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))



if __name__ == "__main__":
    main(sys.argv)
