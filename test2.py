
from keras.layers import Conv2D, UpSampling2D,Dense,Dropout,BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
import os
import pathlib
import pandas as pd 
import keras
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.optimizers import SGD
from keras.optimizers import Adam
import datasetPreparation
import matplotlib.pyplot as plt
import datasetPreparation as dp 
from tensorflow.keras import regularizers
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
path_train = 'images/colorization/color/'
path_test = 'images/validation/'
csv_file1 = r'filenames.csv'
csv_file2 = r'filenames_test.csv'
train_data = dp.load_samples(csv_file1)
test_data = dp.load_samples(csv_file2)
num_train = len(train_data)
num_test = len(test_data)
BATCH_SIZE=24
ds_train =  dp.generator(train_data,batch_size=BATCH_SIZE)
ds_test  = dp.generator(test_data,batch_size=BATCH_SIZE)

# for batch_idx,(X_batch,Y_batch) in enumerate(ds_train):
# 	count =0
# 	for x,y in zip(X_batch,Y_batch):
# 		count+=1
# 		print(x.shape)
# 		print(y.shape)
# 	break
#Convert from RGB to Lab
"""
by iterating on each image, we convert the RGB to Lab. 
Think of LAB image as a grey image in L channel and all color info stored in A and B channels. 
The input to the network will be the L channel, so we assign L channel to X vector. 
And assign A and B to Y.

"""

# l2_reg = l2(1e-3)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=5,
    decay_rate=0.9)

# X_test=[]
# Y_test = []
# count=0
# #restrict values to between -1 and 1		
# for img in images:
# 	try:
# 		count+=1
# 		if count == 1000:
# 			break
# 		lab = rgb2lab(img)
# 		X_test.append(lab[:,:,0])
# 		Y_test.append(lab[:,:,1:]/128)
# 	except:
# 		print("error")	

# X_test = np.array(X_test)
# # Y_test = np.array(Y_test)
# X_test = X_test.reshape(X_test.shape+(1,))
my_callbacks = [
	tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10,  mode="auto",verbose=1)
	]
# #Encoder

# model = Sequential() 

def customLoss(truth,predict):
	loss=tf.keras.losses.MeanSquaredError()
	loss = loss(truth,predict)
	loss = loss * 0.35
	return loss

model = Sequential()
############################ FIRST MODEL #####################################################################
model.add(Conv2D(117, (3, 3), activation='relu', padding='same',strides=2,input_shape=(256, 256, 1)))

model.add(Conv2D(46, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(118, (3,3), activation='relu', padding='same', strides=2))
# model.add(BatchNormalization())

model.add(Conv2D(240, (3,3), activation='relu', padding='same'))
model.add(Conv2D(135, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(37, (3,3), activation='relu', padding='same'))
model.add(Dropout(0.25))
# model.add(BatchNormalization())
model.add(Conv2D(472, (3,3), activation='relu', padding='same'))
model.add(Conv2D(75, (3,3), activation='relu', padding='same'))

#Decoder

model.add(Conv2D(87, (3,3), activation='relu', padding='same'))
# model.add(BatchNormalization())

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(54, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(26, (3,3), activation='relu', padding='same',dilation_rate=(2,2)))
# model.add(BatchNormalization())

model.add(Conv2D(11, (3,3), activation='relu', padding='same',dilation_rate=(2,2)))

model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(BatchNormalization())
##################################################################################################################

############################################# MODEL 2 ############################################################
# #conv1
# model.add(Conv2D(64,(3,3),strides=2,activation='relu',padding='same',input_shape=(256,256,1)))


# #Conv2 
# model.add(Conv2D(128,(3,3),strides=2,activation='relu',padding='same'))
# model.add(Conv2D(128,(3,3),strides=1,activation='relu',padding='same'))



# #conv3
# model.add(Conv2D(256,(3,3),strides=2,activation='relu',padding='same'))
# model.add(Conv2D(256,(3,3),strides=1,activation='relu',padding='same'))



# #Conv4
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same'))
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same'))



# #conv5
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same',dilation_rate=(2,2)))
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same',dilation_rate=(2,2)))



# #conv6
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same',dilation_rate=(2,2)))
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same',dilation_rate=(2,2)))


# # #conv7
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same'))
# model.add(Conv2D(512,(3,3),strides=1,activation='relu',padding='same'))



# #conv8
# model.add(UpSampling2D((2,2)))
# model.add(Conv2D(256,(3,3),strides=1,activation='relu',padding='same'))
# model.add(Conv2D(2,(3,3),strides=1,activation='tanh',padding='same'))
# model.add(UpSampling2D((2,2)))
# model.add(UpSampling2D((2,2)))

####################################################################################################################
# model = tf.keras.models.load_model('other_files/colorize_autoencoder2_custom.model',
#                    custom_objects=None,
#                    compile=False)
model.summary()

model.compile(loss=customLoss,metrics=["accuracy"],optimizer=Adam(lr=0.0001))
# c_train = ct.CustomFit(model)
# c_train.compile(loss='mse',metrics=["accuracy"],optimizer='adam')
# c_train.fit(ds_train,epochs=2,batch_size=32)

model.fit(ds_train,validation_data=ds_test,validation_steps=num_test//BATCH_SIZE,epochs=50,steps_per_epoch=len(train_data)//BATCH_SIZE)

# acc_metric = keras.metrics.Accuracy()
# epochs = 10
# optimizer = Adam(lr=0.0001)
# loss_fn = euclideanLoss

# def train_on_batch(ds_train,BATCH_SIZE):
# 	# model.compile(loss='mse',metrics=["accuracy"],optimizer='adam')
# 	for epoch in range(epochs):
# 		batch=0
# 		print(f"\nstart of Training Epoch {epoch}")
		
# 		for batch_idx,(X_batch,Y_batch) in enumerate(ds_train):
# 			with tf.GradientTape() as tape:
# 				y_pred = model(X_batch,training=True)
# 				loss = loss_fn(Y_batch,y_pred)
# 			gradients = tape.gradient(loss,model.trainable_weights)
# 			optimizer.apply_gradients(zip(gradients,model.trainable_weights))
# 			acc_metric.update_state(Y_batch,y_pred)
# 			# print(acc_metric.result())

# 		# # 	
# 			batch+=1
# 			if batch >= num_train//BATCH_SIZE:
# 				break

# 		train_acc = acc_metric.result()
# 		print("Accuracy over Epoch {}".format(train_acc))
# 		acc_metric.reset_states()
# train_on_batch(ds_train,BATCH_SIZE)
model.save('other_files/colorize_autoencoder2_custom2.model')


