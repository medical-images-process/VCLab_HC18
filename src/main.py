import os
import sys

from src.dataLoader import DataGenerator
from src.models.u_net import get_advanced_unet_512 as get_unet


def main(argv):
    ###############################
    # Parameters                  #
    ###############################
    # training parameter
    num_epochs = 100
    batch_size = 8
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
                                             batch_size= batch_size, transform=imager_transformer, shuffle=shuffle)


    ###############################
    # Start learning              #
    ###############################

    # load model
    print("load model")
    model_template, model = get_unet()

    # train model
    print("train model")
    model.fit_generator(generator=training_generator,
              steps_per_epoch=num_training_samples / batch_size,
              epochs=num_epochs,
              verbose=1,
              shuffle=shuffle)

    # Save model via the template model (which shares the same weights):
    model_template.save(model_path)
    print("Saved model to disk")

    # evaluate the model
    print("evaluate model")
    scores = model.evaluate_generator(training_generator, num_training_samples/batch_size, workers=num_workers)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))



if __name__ == "__main__":
    main(sys.argv)
