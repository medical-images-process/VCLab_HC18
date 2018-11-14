import os
import sys
import keras

from dataLoader import DataGenerator
from models import resnet
from models import u_net

def main(argv):
    # Code main method
    epochs = 100
    params = {'dim': (32, 32, 32),
              'batch_size': 64,
              'n_classes': 6,
              'n_channels': 1,
              'shuffle': True}

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_resnet_trained_model.h5'

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['test'], labels, **params)

    # build model
#    model = u_net.get_unet_128()

    # Train model on dataset
#    model.fit_generator(generator=training_generator,
#                        validation_data=validation_generator,
#                        use_multiprocessing=True,
#                        workers=6)




if __name__ == "__main__":
    main(sys.argv)
