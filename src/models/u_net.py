from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Dense, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Flatten
from keras.optimizers import RMSprop
from src.models import losses
import keras as K
import numpy as np

# ===========================================================================
# \brief Symmetric Unet for shapes 512,512
#
def get_advanced_unet_512(input_shape=(512, 512, 1),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512x512x1

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256x256x16

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128x128x32

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64x64x64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32x32x128

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16x16x256

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8x8x512

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center 8x8x1024

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16x16x512

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32x32x256

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64x64x128

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128x128x64

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256x256x32

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512x512x16

    segmentation = Conv2D(num_classes, (1, 1), activation='softmax')(up0a)
    # Output layer (segmentation) -> 512x512x3

    # prediction of head circumference
    hc4 = Flatten()(center)

    hc3 = Dense(128)(hc4)
    hc3 = BatchNormalization()(hc3)
    hc3 = Activation('relu')(hc3)

    hc2 = Dense(512, activation='relu')(hc3)
    hc2 = BatchNormalization()(hc2)
    hc2 = Activation('relu')(hc2)

    hc1 = Dense(256, activation='relu')(hc2)
    hc1 = BatchNormalization()(hc1)
    hc1 = Activation('relu')(hc1)

    hc00 = Dense(1, activation='softmax')(hc1)
    hc00 = BatchNormalization()(hc00)
    hc00 = Activation('relu')(hc00)
    # Output (center_x) -> 1x1

    hc01 = Dense(1, activation='softmax')(hc1)
    hc01 = BatchNormalization()(hc01)
    hc01 = Activation('relu')(hc01)
    # Output (center_y) -> 1x1

    hc02 = Dense(1, activation='softmax')(hc1)
    hc02 = BatchNormalization()(hc02)
    hc02 = Activation('relu')(hc02)
    # Output (semi_axis_a) -> 1x1

    hc03 = Dense(1, activation='softmax')(hc1)
    hc03 = BatchNormalization()(hc03)
    hc03 = Activation('relu')(hc03)
    # Output (semi_axis_b) -> 1x1

    hc04 = Dense(1, activation='softmax')(hc1)
    hc04 = BatchNormalization()(hc04)
    hc04 = Activation('relu')(hc04)
    # Output (theta) -> 1x1

    hc05 = Dense(1, activation='softmax')(hc1)
    hc05 = BatchNormalization()(hc05)
    hc05 = Activation('relu')(hc05)
    # Output (hc) -> 1x1

    # build model with two outputs
    model = Model(inputs=inputs, outputs=[segmentation, hc00, hc01, hc02, hc03, hc04, hc05])

    try:
        parallel_model = multi_gpu_model(model=model, cpu_relocation=True)
        print("Training using multi GPU...")
    except ValueError:
        parallel_model = model
        print("Training using single GPU oder CPU...")

    class_weights_2 = np.ones(num_classes)
    weighted_loss = losses.class_weighted_cross_entropy_3(class_weights_2)

    parallel_model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_loss, metrics=[losses.dice_coeff])
    return model, parallel_model

