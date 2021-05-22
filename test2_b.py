

import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize,rescale
from skimage.io import imsave, imshow
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import os 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import datasetPreparation as dp 
import testDataGenerator as tdg
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


model = tf.keras.models.load_model('other_files/colorize_autoencoder2.model',
								   custom_objects=None,
								   compile=True)
model.compile(loss='mse',metrics=["accuracy"],optimizer='adam')
# csv_filename = 'filenames_test.csv'
# filenames = dp.load_samples(csv_filename)
# num_files = len(filenames)
# print(num_files)

# ds_test = tdg.generator(filenames,batch_size=18)

# scores = model.evaluate(ds_test,verbose=1,steps=num_files//18)
# print(scores)
def colorize(img_path,saveFile):
	try:
		img1_color = []
		img1=img_to_array(load_img(img_path))
		img2 = resize(img1 ,(256,256))
		img1_color.append(img2)
		img_l_norm = []
		img_l_norm = np.array(img_l_norm,dtype=float)
		img_l_norm =  rgb2lab(1.0/255*img1)[:,:,:]
		print(img_l_norm.shape)
		img1_color = np.array(img1_color, dtype=float)
		img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,:1]
		img1_color = img1_color.reshape(img1_color.shape+(1,))
		#print(img1_color.shape)
		output1 = model.predict(img1_color)
		output1=output1*128
		output1=output1+5.38
		if output1.shape[1:2] != img1.shape[:2]:
			output1 = tf.image.resize(output1,img1.shape[:2],method='bicubic')

		result = np.zeros(img1.shape)
		result[:,:,0] = img_l_norm[:,:,0]
		result[:,:,1:] = output1[0]
		result = resize(result,img1.shape)
		# result = tf.image.resize(result,img1.shape[:2],method='bicubic')
		
		imsave(saveFile+".png", lab2rgb(result))
		return True
	except Exception as e:
		print("Error {}".format(e))
