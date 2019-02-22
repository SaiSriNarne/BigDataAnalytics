import numpy as np #supporting multi-dimensional arrays and matrices
import os #read or write a file
import cv2
import pandas as pd #data manipulation and analysis
from tqdm import tqdm # for  well-established ProgressBar
from random import shuffle #only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.

from ClassficationFuncs import create_train_data, create_test_labeled_data, label_return

LR = 1e-3
MODEL_NAME = 'Caltech100-3Category-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which save

data_dir = 'data\\'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
IMG_SIZE = 299

#list of categories in array format
CATEGORIES = ['accordion','airplanes','anchor']
NUM_CATEGORIES = len(CATEGORIES)
print (NUM_CATEGORIES)

train_data = create_train_data(CATEGORIES,train_dir,IMG_SIZE)

test_data = create_test_labeled_data(CATEGORIES,test_dir,IMG_SIZE)

import tensorflow as tf #used for machine learning applications such as neural networks
import tflearn #modular and transparent deep learning library built on top of Tensorflow
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 12, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
else :
    train = train_data
    test = train_data

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

import matplotlib.pyplot as plt

fig=plt.figure(figsize = (18,10))
for num,data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    str_label=label_return (model_out)
    y.imshow(orig,cmap='gray',interpolation='nearest')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

with open('sample_submission.csv','w') as f:
    f.write('original category,file, predicted category\n')
    for data in test_data:
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        str_label=label_return (model_out)
        file = img_num
        category = str_label
        row = file + "," + category + "\n"
        f.write(row)
