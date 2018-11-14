from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop
from models import losses
import keras as K
import numpy as np

def get_unet_128(input_shape=(128, 128, 3), num_classes=1):

	with K.backend.tf.name_scope('Input'):
		inputs = Input(shape=input_shape)
		# 128
	
	with K.backend.tf.name_scope('Down0'):
		down1 = Conv2D(64, (3, 3), padding='same')(inputs)
		down1 = BatchNormalization()(down1)
		down1 = Activation('relu')(down1)
		down1 = Conv2D(64, (3, 3), padding='same')(down1)
		down1 = BatchNormalization()(down1)
		down1 = Activation('relu')(down1)
		down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
		# 64

	with K.backend.tf.name_scope('Down1'):
		down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
		down2 = BatchNormalization()(down2)
		down2 = Activation('relu')(down2)
		down2 = Conv2D(128, (3, 3), padding='same')(down2)
		down2 = BatchNormalization()(down2)
		down2 = Activation('relu')(down2)
		down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
		# 32

	with K.backend.tf.name_scope('Down2'):
		down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
		down3 = BatchNormalization()(down3)
		down3 = Activation('relu')(down3)
		down3 = Conv2D(256, (3, 3), padding='same')(down3)
		down3 = BatchNormalization()(down3)
		down3 = Activation('relu')(down3)
		down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
		# 16

	with K.backend.tf.name_scope('Down3'):
		down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
		down4 = BatchNormalization()(down4)
		down4 = Activation('relu')(down4)
		down4 = Conv2D(512, (3, 3), padding='same')(down4)
		down4 = BatchNormalization()(down4)
		down4 = Activation('relu')(down4)
		down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
		# 8

	with K.backend.tf.name_scope('Center'):
		center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
		center = BatchNormalization()(center)
		center = Activation('relu')(center)
		center = Conv2D(1024, (3, 3), padding='same')(center)
		center = BatchNormalization()(center)
		center = Activation('relu')(center)
		# center

	with K.backend.tf.name_scope('Up3'):
		up4 = UpSampling2D((2, 2))(center)
		up4 = concatenate([down4, up4], axis=3)
		up4 = Conv2D(512, (3, 3), padding='same')(up4)
		up4 = BatchNormalization()(up4)
		up4 = Activation('relu')(up4)
		up4 = Conv2D(512, (3, 3), padding='same')(up4)
		up4 = BatchNormalization()(up4)
		up4 = Activation('relu')(up4)
		up4 = Conv2D(512, (3, 3), padding='same')(up4)
		up4 = BatchNormalization()(up4)
		up4 = Activation('relu')(up4)
		# 16

	with K.backend.tf.name_scope('Up2'):
		up3 = UpSampling2D((2, 2))(up4)
		up3 = concatenate([down3, up3], axis=3)
		up3 = Conv2D(256, (3, 3), padding='same')(up3)
		up3 = BatchNormalization()(up3)
		up3 = Activation('relu')(up3)
		up3 = Conv2D(256, (3, 3), padding='same')(up3)
		up3 = BatchNormalization()(up3)
		up3 = Activation('relu')(up3)
		up3 = Conv2D(256, (3, 3), padding='same')(up3)
		up3 = BatchNormalization()(up3)
		up3 = Activation('relu')(up3)
		# 32
	
	with K.backend.tf.name_scope('Up1'):
		up2 = UpSampling2D((2, 2))(up3)
		up2 = concatenate([down2, up2], axis=3)
		up2 = Conv2D(128, (3, 3), padding='same')(up2)
		up2 = BatchNormalization()(up2)
		up2 = Activation('relu')(up2)
		up2 = Conv2D(128, (3, 3), padding='same')(up2)
		up2 = BatchNormalization()(up2)
		up2 = Activation('relu')(up2)
		up2 = Conv2D(128, (3, 3), padding='same')(up2)
		up2 = BatchNormalization()(up2)
		up2 = Activation('relu')(up2)
		# 64

	with K.backend.tf.name_scope('UP0'):
		up1 = UpSampling2D((2, 2))(up2)
		up1 = concatenate([down1, up1], axis=3)
		up1 = Conv2D(64, (3, 3), padding='same')(up1)
		up1 = BatchNormalization()(up1)
		up1 = Activation('relu')(up1)
		up1 = Conv2D(64, (3, 3), padding='same')(up1)
		up1 = BatchNormalization()(up1)
		up1 = Activation('relu')(up1)
		up1 = Conv2D(64, (3, 3), padding='same')(up1)
		up1 = BatchNormalization()(up1)
		up1 = Activation('relu')(up1)
		# 128

	with K.backend.tf.name_scope('Classify'):
		classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

	model = Model(inputs=inputs, outputs=classify)
	model.compile(optimizer=RMSprop(lr=0.0001), loss=losses.bce_dice_loss, metrics=[losses.dice_coeff])
	return model
	
# ===========================================================================
# \brief Symmetric Unet sfor shapes 512,512
#
def get_unet_512(input_shape=(512, 512, 3),
				num_classes=1):
	inputs = Input(shape=input_shape)
	# 512
	
	down0a = Conv2D(16, (3, 3), padding='same')(inputs)
	down0a = BatchNormalization()(down0a)
	down0a = Activation('relu')(down0a)
	down0a = Conv2D(16, (3, 3), padding='same')(down0a)
	down0a = BatchNormalization()(down0a)
	down0a = Activation('relu')(down0a)
	down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
	# 256
	
	down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
	down0 = BatchNormalization()(down0)
	down0 = Activation('relu')(down0)
	down0 = Conv2D(32, (3, 3), padding='same')(down0)
	down0 = BatchNormalization()(down0)
	down0 = Activation('relu')(down0)
	down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
	# 128
	
	down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1 = Conv2D(64, (3, 3), padding='same')(down1)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
	# 64
	
	down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2 = Conv2D(128, (3, 3), padding='same')(down2)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
	# 32
	
	down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3 = Conv2D(256, (3, 3), padding='same')(down3)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
	# 16
	
	down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4 = Conv2D(512, (3, 3), padding='same')(down4)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
	# 8
	
	center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	center = Conv2D(1024, (3, 3), padding='same')(center)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	# center
	
	up4 = UpSampling2D((2, 2))(center)
	up4 = concatenate([down4, up4], axis=3)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	# 16
	
	up3 = UpSampling2D((2, 2))(up4)
	up3 = concatenate([down3, up3], axis=3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	# 32
	
	up2 = UpSampling2D((2, 2))(up3)
	up2 = concatenate([down2, up2], axis=3)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	# 64
	
	up1 = UpSampling2D((2, 2))(up2)
	up1 = concatenate([down1, up1], axis=3)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	# 128
	
	up0 = UpSampling2D((2, 2))(up1)
	up0 = concatenate([down0, up0], axis=3)
	up0 = Conv2D(32, (3, 3), padding='same')(up0)
	up0 = BatchNormalization()(up0)
	up0 = Activation('relu')(up0)
	up0 = Conv2D(32, (3, 3), padding='same')(up0)
	up0 = BatchNormalization()(up0)
	up0 = Activation('relu')(up0)
	up0 = Conv2D(32, (3, 3), padding='same')(up0)
	up0 = BatchNormalization()(up0)
	up0 = Activation('relu')(up0)
	# 256
	
	up0a = UpSampling2D((2, 2))(up0)
	up0a = concatenate([down0a, up0a], axis=3)
	up0a = Conv2D(16, (3, 3), padding='same')(up0a)
	up0a = BatchNormalization()(up0a)
	up0a = Activation('relu')(up0a)
	up0a = Conv2D(16, (3, 3), padding='same')(up0a)
	up0a = BatchNormalization()(up0a)
	up0a = Activation('relu')(up0a)
	up0a = Conv2D(16, (3, 3), padding='same')(up0a)
	up0a = BatchNormalization()(up0a)
	up0a = Activation('relu')(up0a)
	# 512
	
	segmentation = Conv2D(num_classes, (1, 1), activation='softmax')(up0a)
	model = Model(inputs=inputs, outputs=segmentation)
	model = multi_gpu_model(model, gpus=2)
		
	class_weights_2 = np.ones(num_classes)	
	weighted_loss = losses.class_weighted_cross_entropy_3(class_weights_2);
	
	model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_loss, metrics=[losses.dice_coeff])
	return model

# ===========================================================================
# \brief 
def get_unet_128_512(input_shape=(128, 512, 3), num_classes=1):    
	inputs = Input(shape=input_shape)
	# 128
	down1 = Conv2D(64, (3, 3), padding='same')(inputs)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1 = Conv2D(64, (3, 3), padding='same')(down1)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
	# 64

	down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2 = Conv2D(128, (3, 3), padding='same')(down2)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
	# 32

	down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3 = Conv2D(256, (3, 3), padding='same')(down3)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
	# 16

	down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4 = Conv2D(512, (3, 3), padding='same')(down4)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
	# 8

	center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	center = Conv2D(1024, (3, 3), padding='same')(center)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	# center

	up4 = UpSampling2D((2, 2))(center)
	up4 = concatenate([down4, up4], axis=3)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	# 16

	up3 = UpSampling2D((2, 2))(up4)
	up3 = concatenate([down3, up3], axis=3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	# 32

	up2 = UpSampling2D((2, 2))(up3)
	up2 = concatenate([down2, up2], axis=3)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	# 64

	up1 = UpSampling2D((2, 2))(up2)
	up1 = concatenate([down1, up1], axis=3)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	# 128

	classify = Conv2D(num_classes, (1, 1), activation='softmax')(up1)
	model = Model(inputs=inputs, outputs=classify)
	model = multi_gpu_model(model, gpus=2)
	
	class_weights_2 = np.ones(num_classes)
	weighted_loss = losses.class_weighted_cross_entropy_3(class_weights_2);
	
	model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_loss, metrics=[losses.dice_coeff])
	return model
	
# ===========================================================================
# \brief 
def get_unet_128_512_weighted(input_shape=(128, 512, 3), num_classes=1):    
	inputs = Input(shape=input_shape)
	# 128
	down1 = Conv2D(64, (3, 3), padding='same')(inputs)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1 = Conv2D(64, (3, 3), padding='same')(down1)
	down1 = BatchNormalization()(down1)
	down1 = Activation('relu')(down1)
	down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
	# 64

	down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2 = Conv2D(128, (3, 3), padding='same')(down2)
	down2 = BatchNormalization()(down2)
	down2 = Activation('relu')(down2)
	down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
	# 32

	down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3 = Conv2D(256, (3, 3), padding='same')(down3)
	down3 = BatchNormalization()(down3)
	down3 = Activation('relu')(down3)
	down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
	# 16

	down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4 = Conv2D(512, (3, 3), padding='same')(down4)
	down4 = BatchNormalization()(down4)
	down4 = Activation('relu')(down4)
	down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
	# 8

	center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	center = Conv2D(1024, (3, 3), padding='same')(center)
	center = BatchNormalization()(center)
	center = Activation('relu')(center)
	# center

	up4 = UpSampling2D((2, 2))(center)
	up4 = concatenate([down4, up4], axis=3)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	up4 = Conv2D(512, (3, 3), padding='same')(up4)
	up4 = BatchNormalization()(up4)
	up4 = Activation('relu')(up4)
	# 16

	up3 = UpSampling2D((2, 2))(up4)
	up3 = concatenate([down3, up3], axis=3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	up3 = Conv2D(256, (3, 3), padding='same')(up3)
	up3 = BatchNormalization()(up3)
	up3 = Activation('relu')(up3)
	# 32

	up2 = UpSampling2D((2, 2))(up3)
	up2 = concatenate([down2, up2], axis=3)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	up2 = Conv2D(128, (3, 3), padding='same')(up2)
	up2 = BatchNormalization()(up2)
	up2 = Activation('relu')(up2)
	# 64

	up1 = UpSampling2D((2, 2))(up2)
	up1 = concatenate([down1, up1], axis=3)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	up1 = Conv2D(64, (3, 3), padding='same')(up1)
	up1 = BatchNormalization()(up1)
	up1 = Activation('relu')(up1)
	# 128

	classify = Conv2D(num_classes, (1, 1), activation='softmax')(up1)
	model = Model(inputs=inputs, outputs=classify)
	model = multi_gpu_model(model, gpus=2)
	
	w_background 	= 0.1# background
	w_carotid 		= 1 # carotid artery
	w_jugular 		= 1 # jugular vein
	w_facialis 		= 1 # facial nerve
	w_cochlea 		= 1 # cochlea
	w_chorda		= 10 # chorda
	w_ossicles		= 1 # ossicles
	w_ssc			= 1 # semicircular canals
	w_iac			= 1 # internal auditory canal
	w_aqueduct		= 1 # vestibular aqueduct
	w_eac  			= 1 # external auditory canal
	class_weights_2 = np.ones(num_classes)
	class_weights_2[0] = w_background
	class_weights_2[1] = w_chorda
	weighted_loss = losses.class_weighted_cross_entropy_3(class_weights_2);
	
	model.compile(optimizer=RMSprop(lr=0.0001), loss=weighted_loss, metrics=[losses.dice_coeff])
	return model