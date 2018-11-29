import os
import sys
import keras

from src.models.u_net import get_unet_512
from src.dataLoader import DataGenerator


def main(argv):
    ###############################
    # Parameters                  #
    ###############################
    # training parameter
    num_epochs = 100
    batch_size = 64
    num_workers = 4
    shuffle = False
    num_training_samples = 999

    # paths
    train_dataset_dir = os.path.join(os.getcwd(),"Dataset", "training")
    train_csv = os.path.join(train_dataset_dir, 'training_set_pixel_size_and_HC.csv')

    test_dataset_dir = os.path.join(os.getcwd(), 'Dataset/test')
    test_csv = os.path.join(train_dataset_dir, 'test_set_pixel_size.csv')

    model_path = os.path.join(os.getcwd(), 'saved_models/keras_unet_trained_model.h5')

    ###############################
    # Load Dataset                #
    ###############################
    print("load dataset")
    imager_transformer = {'reshape': 512}
    training_generator = DataGenerator(csv_file=train_csv, root_dir=os.path.join(train_dataset_dir, 'set'),
                                             batch_size= batch_size, transform=imager_transformer)

    ###############################
    # Start learning              #
    ###############################

    # load model
    print("load model")
    model = get_unet_512()

    # train model
    print("train model")
    model.fit_generator(generator=training_generator,
              steps_per_epoch=num_training_samples / batch_size,
              epochs=num_epochs,
              verbose=1,
              shuffle=shuffle)


    # evaluate the model
    print("evaluate model")
    scores = model.evaluate_generator(training_generator, num_training_samples/batch_size, workers=num_workers)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_path)
    print("Saved model to disk")



if __name__ == "__main__":
    main(sys.argv)
