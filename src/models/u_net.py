import numpy as np
from src.models import losses
from keras import optimizers
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers import Input, concatenate, Conv2D, Dense, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU


# ===========================================================================
# \brief Symmetric Unet for shapes 512,512
#
def create_unet_512x512(input_shape=(512, 512, 1), pooling_mode='avg',
                        num_classes=1):
    inputs = Input(shape=input_shape)
    # 512x512x1

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU(alpha=0.1)(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU(alpha=0.1)(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256x256x16

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128x128x32

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64x64x64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32x32x128

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU(alpha=0.1)(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16x16x256

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU(alpha=0.1)(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8x8x512

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    # center 8x8x1024

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU(alpha=0.1)(up4)
    # 16x16x512

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU(alpha=0.1)(up3)
    # 32x32x256

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    # 64x64x128

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    # 128x128x64

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    # 256x256x32

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU(alpha=0.1)(up0a)
    # 512x512x16

    segmentation = Conv2D(num_classes, (1, 1), activation='softmax')(up0a)
    # Output layer (segmentation) -> 512x512x3

    # prediction of ellipse parameters
    if pooling_mode == 'flatten':
        from keras.layers import Flatten
        ep4 = Flatten()(center)
    if pooling_mode == 'avg':
        from keras.layers import GlobalAveragePooling2D
        ep4 = GlobalAveragePooling2D()(center)
    if pooling_mode == 'max':
        from keras.layers import GlobalMaxPooling2D
        ep4 = GlobalMaxPooling2D()(center)

    ep3 = Dense(128)(ep4)
    ep3 = BatchNormalization()(ep3)
    ep3 = LeakyReLU(alpha=0.1)(ep3)

    ep2 = Dense(512)(ep3)
    ep2 = BatchNormalization()(ep2)
    ep2 = LeakyReLU(alpha=0.1)(ep2)

    ep1 = Dense(256)(ep2)
    ep1 = BatchNormalization()(ep1)
    ep1 = LeakyReLU(alpha=0.1)(ep1)

    ep_center_x = Dense(1, activation='relu')(ep1)
    # Output (center_x) -> 1x1

    ep_center_y = Dense(1, activation='relu')(ep1)
    # Output (center_y) -> 1x1

    ep_axis_a = Dense(1, activation='relu')(ep1)
    # Output (semi_axis_a) -> 1x1

    ep_axis_b = Dense(1, activation='relu')(ep1)
    # Output (semi_axis_b) -> 1x1

    ep_angle_sin = Dense(1, activation='tanh')(ep1)
    # Output (angle as sin) -> 1x1

    ep_angle_cos = Dense(1, activation='tanh')(ep1)
    # Output (angle as cos) -> 1x1

    ep_hc = Dense(1, activation='relu')(ep1)
    # Output (hc) -> 1x1

    # build model with two outputs
    model = Model(inputs=inputs,
                  outputs=[segmentation, ep_center_x, ep_center_y, ep_axis_a, ep_axis_b, ep_angle_sin, ep_angle_cos,
                           ep_hc])
    return model


# ===========================================================================
# \brief Symmetric Unet for shapes 512,512 with minimized numbers of parameter
#
def create_unet_min_512x512(input_shape=(512, 512, 1), pooling_mode='avg',
                            num_classes=1):
    inputs = Input(shape=input_shape)
    # 512x512x1

    down0 = Conv2D(16, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(16, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(16, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(4, 4))(down0)
    # 128x128x32

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1 = Conv2D(128, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(4, 4))(down1)
    # 32x32x128

    down2 = Conv2D(256, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(4, 4))(down2)
    # 8x8x512

    center = Conv2D(1024, (3, 3), padding='same')(down2_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    # center 8x8x1024

    up2 = UpSampling2D((4, 4))(center)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(256, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    # 32x32x256

    up1 = UpSampling2D((4, 4))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    # 128x128x64

    up0 = UpSampling2D((4, 4))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    # 512x512x16

    segmentation = Conv2D(num_classes, (1, 1), activation='softmax')(up0)
    # Output layer (segmentation) -> 512x512x3

    # prediction of ellipse parameters
    if pooling_mode == 'flatten':
        from keras.layers import Flatten
        ep4 = Flatten()(center)
    if pooling_mode == 'avg':
        from keras.layers import GlobalAveragePooling2D
        ep4 = GlobalAveragePooling2D()(center)
    if pooling_mode == 'max':
        from keras.layers import GlobalMaxPooling2D
        ep4 = GlobalMaxPooling2D()(center)

    ep3 = Dense(128)(ep4)
    ep3 = BatchNormalization()(ep3)
    ep3 = LeakyReLU(alpha=0.1)(ep3)

    ep2 = Dense(512)(ep3)
    ep2 = BatchNormalization()(ep2)
    ep2 = LeakyReLU(alpha=0.1)(ep2)

    ep1 = Dense(256)(ep2)
    ep1 = BatchNormalization()(ep1)
    ep1 = LeakyReLU(alpha=0.1)(ep1)

    ep_center_x = Dense(1, activation='relu')(ep1)
    # Output (center_x) -> 1x1

    ep_center_y = Dense(1, activation='relu')(ep1)
    # Output (center_y) -> 1x1

    ep_axis_a = Dense(1, activation='relu')(ep1)
    # Output (semi_axis_a) -> 1x1

    ep_axis_b = Dense(1, activation='relu')(ep1)
    # Output (semi_axis_b) -> 1x1

    ep_angle_sin = Dense(1, activation='tanh')(ep1)
    # Output (angle as sin) -> 1x1

    ep_angle_cos = Dense(1, activation='tanh')(ep1)
    # Output (angle as cos) -> 1x1

    ep_hc = Dense(1, activation='relu')(ep1)
    # Output (hc) -> 1x1

    # build model with two outputs
    model = Model(inputs=inputs,
                  outputs=[segmentation, ep_center_x, ep_center_y, ep_axis_a, ep_axis_b, ep_angle_sin, ep_angle_cos,
                           ep_hc])
    return model


# ===========================================================================
# \brief Symmetric Unet for shapes 800,540 with minimized numbers of parameter
#
def create_unet_min_540x800(input_shape=(540, 800, 1), pooling_mode='avg',
                            num_classes=1):
    inputs = Input(shape=input_shape)
    # 800x540x1

    down0 = Conv2D(16, (3, 3), padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(16, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0 = Conv2D(16, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU(alpha=0.1)(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(6, 4))(down0)
    # 128x128x32

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1 = Conv2D(128, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU(alpha=0.1)(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(6, 4))(down1)
    # 32x32x128

    down2 = Conv2D(256, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU(alpha=0.1)(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(3, 2))(down2)
    # 8x8x512

    center = Conv2D(1024, (3, 3), padding='same')(down2_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU(alpha=0.1)(center)
    # center 8x8x1024

    up2 = UpSampling2D((3, 2))(center)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(256, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU(alpha=0.1)(up2)
    # 32x32x256

    up1 = UpSampling2D((6, 4))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU(alpha=0.1)(up1)
    # 128x128x64

    up0 = UpSampling2D((6, 4))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU(alpha=0.1)(up0)
    # 512x512x16

    segmentation = Conv2D(num_classes, (1, 1), activation='softmax', bias_initializer='zeros')(up0)
    # Output layer (segmentation) -> 512x512x3

    # prediction of ellipse parameters
    if pooling_mode == 'flatten':
        from keras.layers import Flatten
        ep4 = Flatten()(center)
    if pooling_mode == 'avg':
        from keras.layers import GlobalAveragePooling2D
        ep4 = GlobalAveragePooling2D()(center)
    if pooling_mode == 'max':
        from keras.layers import GlobalMaxPooling2D
        ep4 = GlobalMaxPooling2D()(center)

    ep3 = Dense(128)(ep4)
    ep3 = BatchNormalization()(ep3)
    ep3 = LeakyReLU(alpha=0.1)(ep3)

    ep2 = Dense(512)(ep3)
    ep2 = BatchNormalization()(ep2)
    ep2 = LeakyReLU(alpha=0.1)(ep2)

    ep1 = Dense(256)(ep2)
    ep1 = BatchNormalization()(ep1)
    ep1 = LeakyReLU(alpha=0.1)(ep1)

    ep_center_x = Dense(1, activation='relu', bias_initializer='zeros')(ep1)
    # Output (center_x) -> 1x1

    ep_center_y = Dense(1, activation='relu', bias_initializer='zeros')(ep1)
    # Output (center_y) -> 1x1

    ep_axis_a = Dense(1, activation='relu', bias_initializer='zeros')(ep1)
    # Output (semi_axis_a) -> 1x1

    ep_axis_b = Dense(1, activation='relu', bias_initializer='zeros')(ep1)
    # Output (semi_axis_b) -> 1x1

    ep_angle_sin = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # Output (angle as sin) -> 1x1

    ep_angle_cos = Dense(1, activation='tanh', bias_initializer='zeros')(ep1)
    # Output (angle as cos) -> 1x1

    ep_hc = Dense(1, activation='relu', bias_initializer='zeros')(ep1)
    # Output (hc) -> 1x1

    # build model with two outputs
    model = Model(inputs=inputs,
                  outputs=[segmentation, ep_center_x, ep_center_y, ep_axis_a, ep_axis_b, ep_angle_sin, ep_angle_cos,
                           ep_hc])
    return model


# ===========================================================================
# \brief Returns a symmetric Unet model arcitecture
#
def get_unet(model_name='unet_opt_512x512', input_shape=(512, 512, 1), pooling_mode='avg', num_classes=1, lr=0.01):
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

    class_weights_2 = np.ones(num_classes)
    weighted_loss = losses.class_weighted_cross_entropy_3(class_weights_2)

    optimizer = optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)
    parallel_model.compile(loss='mse', loss_weights=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], optimizer=optimizer, metrics=['mse'])
    # parallel_model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model, parallel_model
