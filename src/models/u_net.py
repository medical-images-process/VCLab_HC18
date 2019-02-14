import numpy as np
from models import losses
from keras import optimizers
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers import Input, concatenate, Conv2D, Dense, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU, \
    Dropout


def create_unet_256x384(input_shape=(256, 384, 1), pooling_mode='avg',
                        num_classes=1):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    # ###########################################
    # prediction of ellipse parameters
    if pooling_mode == 'flatten':
        from keras.layers import Flatten
        ep4 = Flatten()(conv5)
    if pooling_mode == 'avg':
        from keras.layers import GlobalAveragePooling2D
        ep4 = GlobalAveragePooling2D()(conv5)
    if pooling_mode == 'max':
        from keras.layers import GlobalMaxPooling2D
        ep4 = GlobalMaxPooling2D()(conv5)

    # ##########
    ep3 = Dense(128, activation='relu', kernel_initializer='he_normal')(ep4)
    # ##########
    ep2 = Dense(512, activation='relu', kernel_initializer='he_normal')(ep3)
    # ##########
    ep1 = Dense(256, activation='relu', kernel_initializer='he_normal')(ep2)
    # ##########
    ep_center_x = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (center_x) -> 1x1
    # ##########
    ep_center_y = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (center_y) -> 1x1
    # ##########
    ep_axis_a = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (semi_axis_a) -> 1x1
    # ##########
    ep_axis_b = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (semi_axis_b) -> 1x1
    # ##########
    ep_angle_sin = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (angle as sin) -> 1x1
    # ##########
    ep_angle_cos = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (angle as cos) -> 1x1
    # ##########
    ep_hc = Dense(1, activation='tanh', bias_initializer='he_normal')(ep1)
    # Output (hc) -> 1x1
    # ##########
    model = Model(input=inputs,
                  output=conv10)
    model = Model(input=inputs,
                  output=[conv10, ep_center_x, ep_center_y, ep_axis_a, ep_axis_b, ep_angle_sin, ep_angle_cos, ep_hc])

    return model


# ===========================================================================
# \brief Symmetric Unet for shapes 800,540 with minimized numbers of parameter
#
def create_unet_min_128x192(input_shape=(128, 192, 1), pooling_mode='avg',
                            num_classes=1):
    # ##########
    inputs = Input(shape=input_shape)
    # ##########
    # 135x200x1
    down0 = Conv2D(16, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)

    down0 = Conv2D(16, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)

    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)

    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # ##########
    # 64x96x32
    # ##########
    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)

    down1 = Conv2D(128, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # ##########
    # 32x48x64
    # ##########
    down2 = Conv2D(256, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)

    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # ##########
    # 16x24x256
    # ##########
    center = Conv2D(512, (3, 3), padding='same')(down2_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    # ##########
    # center 16x24x512
    # ##########
    up2 = UpSampling2D((2, 2))(center)
    up2 = concatenate([down2, up2], axis=3)

    up2 = Conv2D(256, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    # ##########
    # 32x48x256
    # ##########
    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)

    up1 = Conv2D(128, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)

    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    # ##########
    # 64x92x64
    # ##########
    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)

    up0 = Conv2D(16, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)

    up0 = Conv2D(16, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)

    up0 = Conv2D(16, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    # ##########
    # 128x192x32
    # ##########
    segmentation = Conv2D(num_classes, (1, 1), activation='sigmoid', bias_initializer='zeros')(up0)
    # Output layer (segmentation) -> 128x192x1

    # # ###########################################
    # # prediction of ellipse parameters
    # if pooling_mode == 'flatten':
    #     from keras.layers import Flatten
    #     ep4 = Flatten()(center)
    # if pooling_mode == 'avg':
    #     from keras.layers import GlobalAveragePooling2D
    #     ep4 = GlobalAveragePooling2D()(center)
    # if pooling_mode == 'max':
    #     from keras.layers import GlobalMaxPooling2D
    #     ep4 = GlobalMaxPooling2D()(center)
    #
    # # ##########
    # ep3 = Dense(128)(ep4)
    # ep3 = BatchNormalization()(ep3)
    # ep3 = LeakyReLU(alpha=0.1)(ep3)
    # # ##########
    # ep2 = Dense(512)(ep3)
    # ep2 = BatchNormalization()(ep2)
    # ep2 = LeakyReLU(alpha=0.1)(ep2)
    # # ##########
    # ep1 = Dense(256)(ep2)
    # ep1 = BatchNormalization()(ep1)
    # ep1 = LeakyReLU(alpha=0.1)(ep1)
    # # ##########
    # ep_center_x = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (center_x) -> 1x1
    # # ##########
    # ep_center_y = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (center_y) -> 1x1
    # # ##########
    # ep_axis_a = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (semi_axis_a) -> 1x1
    # # ##########
    # ep_axis_b = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (semi_axis_b) -> 1x1
    # # ##########
    # ep_angle_sin = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (angle as sin) -> 1x1
    # # ##########
    # ep_angle_cos = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (angle as cos) -> 1x1
    # # ##########
    # ep_hc = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # # Output (hc) -> 1x1
    # # ##########

    # build model with two outputs
    model = Model(inputs=inputs,
                  outputs=segmentation)

    return model


# ===========================================================================
# \brief Returns a symmetric Unet model arcitecture
#
def get_unet(model_name='unet_256x256', input_shape=(256, 256, 1), pooling_mode='avg', num_classes=1, lr=1e-4):
    print('Create model: ' + model_name)
    model = globals()['create_' + model_name](input_shape=input_shape, pooling_mode=pooling_mode,
                                              num_classes=num_classes)

    # parallelize model
    try:
        parallel_model = multi_gpu_model(model=model, cpu_relocation=False)
        print("Training using multi GPU...")
    except ValueError:
        parallel_model = model
        print("Training using single GPU oder CPU...")

    weights = np.ones(1)
    weights[0] = 1
    loss = ['binary_crossentropy', 'mse', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae']

    parallel_model.compile(optimizer=optimizers.Adam(lr=lr), loss=loss, metrics=['accuracy'])

    return model, parallel_model
