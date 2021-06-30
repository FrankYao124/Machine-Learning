# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import tensorflow as tf 
from matplotlib import pyplot as plt
from keras.models import save_model, load_model, Model
from keras.layers import Input, Dropout, BatchNormalization, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config=tf.compat.v1.ConfigProto()
sess=tf.compat.v1.Session(config=config) 
#config = tf.ConfigProto()
config.allow_soft_placement=True 
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
input_name = os.listdir('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/completeData/left_images/')
input_name1 = os.listdir('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/testData/left_images/') 
n = len(input_name)
batch_size = 8
input_size_1 = 256
input_size_2 = 256
"""
Batch_data
"""
def batch_data(input_name, n, batch_size = 8, input_size_1 = 256, input_size_2 = 256):
    rand_num = random.randint(0, n-1)
    img1 = io.imread('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/completeData/left_images/'+input_name[rand_num]).astype("float")
    img2 = io.imread('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth/'+input_name[rand_num]).astype("float")
    img1 = resize(img1, [input_size_1, input_size_2, 3])
    img2 = resize(img2, [input_size_1, input_size_2, 3])
    img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
    img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
    img1 /= 255
    img2 /= 255
    batch_input = img1
    batch_output = img2
    for batch_iter in range(1, batch_size):
        rand_num = random.randint(0, n-1)
        img1 = io.imread('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/completeData/left_images/'+input_name[rand_num]).astype("float")
        img2 = io.imread('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/completeData/left_groundTruth/'+input_name[rand_num]).astype("float")
        img1 = resize(img1, [input_size_1, input_size_2, 3])
        img2 = resize(img2, [input_size_1, input_size_2, 3])
        img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
        img2 = np.reshape(img2, (1, input_size_1, input_size_2, 3))
        img1 /= 255
        img2 /= 255
        batch_input = np.concatenate((batch_input, img1), axis = 0)
        batch_output = np.concatenate((batch_output, img2), axis = 0)
    return batch_input, batch_output
 
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
 
def Conv2dT_BN(x, filters, kernel_size, strides=(2,2), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
 
inpt = Input(shape=(input_size_1, input_size_2, 3))
x = Conv2d_BN(inpt, 4, (3, 3))
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Conv2d_BN(x, 8, (3, 3))
x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Conv2d_BN(x, 16, (3, 3))
x = AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Conv2d_BN(x, 32, (3, 3))
x = AveragePooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
x = Conv2d_BN(x, 64, (3, 3))
x = Dropout(0.5)(x)
x = Conv2d_BN(x, 64, (1, 1))
x = Dropout(0.5)(x)
x = Conv2dT_BN(x, 32, (3, 3))
x = Conv2dT_BN(x, 16, (3, 3))
x = Conv2dT_BN(x, 8, (3, 3))
x = Conv2dT_BN(x, 4, (3, 3))
 
x = Conv2DTranspose(filters=3,kernel_size=(3,3),strides=(1,1),padding='same',activation='sigmoid')(x)
 
model = Model(inpt, x)
model.summary()
 
model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])
itr = 1000
S = []
for i in range(itr):
    print("iteration = ", i+1)
    if i < 500:
        bs = 4
    elif i < 2000:
        bs = 8
    elif i < 5000:
        bs = 16
    else:
        bs = 32
    train_X, train_Y = batch_data(input_name, n, batch_size = bs)
    model.fit(train_X, train_Y, epochs=1, verbose=0)
#-------------------
img1 = io.imread('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/testData/left_images/'+input_name1[15]).astype("float")
img1 = resize(img1, [input_size_1, input_size_2, 3])
img1 = np.reshape(img1, (1, input_size_1, input_size_2, 3))
img1 /= 255
img2 = model.predict(img1)
img1 = np.reshape(img1, ( input_size_1, input_size_2, 3))
plt.axis('off')
plt.figure()
plt.imshow(img1)
img2 = np.reshape(img2, ( input_size_1, input_size_2, 3))
plt.axis('off')
plt.figure()
plt.imshow(img2)
img3 = io.imread('C:/Users/86176/Desktop/PedCut2013_SegmentationDataset/data/testData/left_groundTruth/'+input_name1[15]).astype("float")
img3 = resize(img3, [input_size_1, input_size_2, 3])
img3 = np.reshape(img3, ( input_size_1, input_size_2, 3))
img3 /= 255
plt.axis('off')
plt.figure()
plt.imshow(img3)