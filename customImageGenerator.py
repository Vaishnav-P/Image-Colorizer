import os

import numpy as np
import pandas as pd

from PIL import Image
import matplotlib.pyplot as plt


import cv2
from sklearn.utils import shuffle

from keras.utils import np_utils

path_test = r'images/colorization/color/'

def preparecsv(path):

	data_dir_list = os.listdir(path)
	num_files = len(data_dir_list)
	print(num_files)
	train_df = pd.DataFrame(columns=['Filename'])
	test_df = pd.DataFrame(columns=['Filename'])

	
	img_names= []
	img_filename = []
	for dataset in data_dir_list:
			
		img_filename = os.path.join(path,dataset)

		train_df = train_df.append({'Filename':img_filename},ignore_index=True)	

	train_df.to_csv('other_files/filename.csv')

preparecsv(path_test)