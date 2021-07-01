
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
from matplotlib.colors import LogNorm
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt  
import datasetPreparation as dp 

path_test = 'images/validation/'

csv_file1 = r'filenames_test.csv'
test_data = dp.load_samples(csv_file1)
num_test = len(test_data)
BATCH_SIZE=100
ds_test = dp.generator(test_data,batch_size=BATCH_SIZE)
a = []
b =[]

for batch_idx,(X_batch,Y_batch) in enumerate(ds_test):
	count =0
	for x,y in zip(X_batch,Y_batch):
		a.append(y[:,:,0])
		b.append(y[:,:,1])
	break

a = np.array(a)
print(a.shape)
b = np.array(b)
print(np.max(a),np.min(a))
print(np.max(b),np.min(b))
a = a.ravel()
b = b.ravel()
ab = np.vstack((a,b)).T
#np.arange(-110,120,10)
hist,x,y = np.histogram2d(ab[:,0],ab[:,1],bins=[np.arange(-110,120,10),np.arange(-110,120,10)])
# plt.xlim([-110, 110])
# plt.ylim([-110, 110])
plt.imshow(hist,cmap='plasma',interpolation='nearest')
plt.show()
print(x)
print(y)
print(hist.shape)
# plt.clf()
# plt.close()
# cord = []
# count=0
# shape = hist.shape
# shape_x = shape[0]
# shape_y = shape[1]
# for i in range(shape_x):
# 	for j in range(shape_y):
# 		if hist[i][j] > 0:
# 			print(hist[i][j])
# 			cord.append([i,j])

# print(cord)
# print(len(cord))
# print(hist)
binCentersX = np.sqrt(x[1:]*x[:-1])
print(binCentersX)
binCentersY = np.sqrt(y[1:]*y[:-1])
print(binCentersY)


# for k in hist[:]:
# 	for q in k:
# 		if q > 0:
# 			print(q)
# dis_points = []
# for xx in x:
# 	for yy in y:
# 		dis_points.append([xx,yy])


# for x in dis_points:
# 	print(x)	
# def quantize_image_to_grid_lab(image, grid_size):
    
#     l_channel = lab_image[:, :, 0]
#     ab_channels = lab_image[:, :, 1:]
#     ab_channels = np.round(ab_channels)
#     ab_channels = ab_channels - (ab_channels % grid_size)
#     lab_image[:, :, 1:] = ab_channels
#     return lab_image


# def quantize_image_to_grid(image, grid_size):
#     return lab2rgb(quantize_image_to_grid_lab(image, grid_size))


# img = quantize_image_to_grid(image,10)
# imsave('test_quant.png',img,check_contrast=True)