import os
import cv2
import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle	
import matplotlib.pyplot as plt 
from skimage.transform import resize,rescale
from skimage.color import rgb2lab, lab2rgb
from keras.preprocessing.image import img_to_array, load_img

def preProcessing(img):

	img = resize(img,(256,256))
	img = img/255.0
	lab = rgb2lab(img)
	return (lab[:,:,0],lab[:,:,1:]/128)



def generator(filenames,batch_size=32,shuffle_data=True,resize=256):
	
	num_files = len(filenames)
	count = 0 
	img_count = 0
	while True:

		file = shuffle(filenames)

		for offset in range(0,num_files,batch_size):
			count+=1
			batch_samples = file[offset:offset+batch_size]
			# print("Batch {} batch_size {}".format(count,len(batch_samples)))
		
			X_train = []
			Y_train = []
			for batch_sample in batch_samples:
				try:
							
					img_count+=1	
					img = img_to_array(load_img(batch_sample))
					l,ab = preProcessing(img)
					X_train.append(l)
					Y_train.append(ab)
					
					# print("loaded img {} count {}".format(batch_sample,img_count))
				except Exception as e:
					print("Error {}".format(e))
					continue	

			X_train = np.array(X_train)
			X_train=X_train.reshape(X_train.shape+(1,))
			Y_train = np.array(Y_train)		
	
			yield (X_train,Y_train)	