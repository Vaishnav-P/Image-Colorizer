
from keras.layers import Conv2D, UpSampling2D,Dense,Dropout,BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
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
from tensorflow.python.keras import backend 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.optimizers import SGD
from keras.optimizers import Adam
import datasetPreparation
import matplotlib.pyplot as plt
import datasetPreparation as dp 
from keras.regularizers import l2
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
sess = backend.get_session()
path_train = 'images/colorization/color/'
path_test = 'images/validation/'
csv_file = r'filenames.csv'

train_data = dp.load_samples(csv_file)

num_train = len(train_data)
BATCH_SIZE=18
ds_train =  dp.generator(train_data,batch_size=BATCH_SIZE)


# for batch_idx,(X_batch,Y_batch) in enumerate(ds_train):
# 	count =0
# 	for x,y in zip(X_batch,Y_batch):
# 		count+=1
# 		print(x.shape)
# 		print(y.shape)
# 	break


l2_reg = l2(1e-3)

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
	tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2,  mode="auto",verbose=1)
	]
# #Encoder

model = Sequential() 
model.add(Conv2D(117, (3, 3), activation='relu', padding='same',strides=2,input_shape=(256, 256, 1)))
model.add(Conv2D(46, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(118, (3,3), activation='relu', padding='same', strides=2))
model.add(BatchNormalization())
model.add(Conv2D(240, (3,3), activation='relu', padding='same'))
model.add(Conv2D(135, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(37, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(472, (3,3), activation='relu', padding='same'))
model.add(Conv2D(75, (3,3), activation='relu', padding='same'))
#Decoder

model.add(Conv2D(87, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(54, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same',dilation_rate=(2,2)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(26, (3,3), activation='relu', padding='same',dilation_rate=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(11, (3,3), activation='relu', padding='same',dilation_rate=(2,2)))

model.add(Conv2D(2, (3, 3), activation='tanh', padding='same',dilation_rate=(2,2)))
model.add(UpSampling2D((2, 2)))
model.add(BatchNormalization())
model.summary()

model.compile(loss='mse',metrics=["accuracy"],optimizer=Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0,epsilon=1e-8))
# c_train = ct.CustomFit(model)
# c_train.compile(loss='mse',metrics=["accuracy"],optimizer='adam')
# c_train.fit(ds_train,epochs=2,batch_size=32)

model.fit(ds_train,epochs=100,steps_per_epoch=len(train_data)//BATCH_SIZE)

# acc_metric = keras.metrics.Accuracy()
# epochs = 10
# optimizer = Adam(lr=0.001)
# loss_fn = keras.losses.MeanSquaredError()

# def train_on_batch(ds_train,BATCH_SIZE):
# 	# model.compile(loss='mse',metrics=["accuracy"],optimizer='adam')
# 	for epoch in range(epochs):
# 		batch=0
# 		print(f"\nstart of Training Epoch {epoch}")
		
# 		for batch_idx,(X_batch,Y_batch) in enumerate(ds_train):
# 			with tf.GradientTape() as tape:
# 				y_pred = model(X_batch,training=True)
# 				print(y_pred.shape)
# 				loss = loss_fn(Y_batch,y_pred)
# 				print(loss)
# 			gradients = tape.gradient(loss,model.trainable_weights)
# 			optimizer.apply_gradients(zip(gradients,model.trainable_weights))
# 			acc_metric.update_state(Y_batch,y_pred)
# 			print(acc_metric.result())

# 		# # 	
# 			batch+=1
# 			if batch >= num_train//BATCH_SIZE:
# 				break

# 		train_acc = acc_metric.result()
# 		print("Accuracy over Epoch {}".format(train_acc))
# 		acc_metric.reset_states()
# train_on_batch(ds_train,BATCH_SIZE)
model.save('other_files/colorize_autoencoder2.model')



# tf.keras.models.load_model(
# 	'other_files/colorize_autoencoder300.model',
# 	custom_objects=None,
# 	compile=True)

# img1_color=[]

# img1=img_to_array(load_img('images/sunset.png'))
# img1 = resize(img1 ,(256,256))
# img1_color.append(img1)

# img1_color = np.array(img1_color, dtype=float)
# img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
# img1_color = img1_color.reshape(img1_color.shape+(1,))

# output1 = model.predict(img1_color)
# output1 = output1*128

# result = np.zeros((256, 256, 3))
# result[:,:,0] = img1_color[0][:,:,0]
# result[:,:,1:] = output1[0]
# imsave("result.png", lab2rgb(result))
# """