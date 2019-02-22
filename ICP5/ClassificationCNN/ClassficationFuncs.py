import numpy as np #supporting multi-dimensional arrays and matrices
import os #read or write a file
import cv2
import pandas as pd #data manipulation and analysis
from tqdm import tqdm # for  well-established ProgressBar
from random import shuffle #only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.
import tensorflow as tf


''' function that accept category and return array format of the vlaue , one-hot array
 am sure there's better way to do this .......'''

def label_img(word_label):
    if word_label == 'accordion': return [1,0,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'airplanes': return [0,1,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'anchor': return [0,0,1,0,0,0,0,0,0,0,0,0]
    # elif word_label == 'Common Chickweed': return [0,0,0,1,0,0,0,0,0,0,0,0]
    # elif word_label == 'Common wheat': return [0,0,0,0,1,0,0,0,0,0,0,0]
    # elif word_label == 'Fat Hen': return [0,0,0,0,0,1,0,0,0,0,0,0]
    # elif word_label == 'Loose Silky-bent': return [0,0,0,0,0,0,1,0,0,0,0,0]
    # elif word_label == 'Maize': return [0,0,0,0,0,0,0,1,0,0,0,0]
    # elif word_label == 'Scentless Mayweed': return [0,0,0,0,0,0,0,0,1,0,0,0]
    # elif word_label == 'Shepherds Purse': return [0,0,0,0,0,0,0,0,0,1,0,0]
    # elif word_label == 'Small-flowered Cranesbill': return [0,0,0,0,0,0,0,0,0,0,1,0]
    # elif word_label == 'Sugar beet': return [0,0,0,0,0,0,0,0,0,0,0,1]

'''function that will create train data , will go thought all the file do this 
----read the image in  grayscale mode ,resize it
---change it to numpy arrays and  append it to dataframe train with it`s associated category '''

def create_train_data(CATEGORIES,train_dir,IMG_SIZE):
    train = []
    for category_id, category in enumerate(CATEGORIES):
        for img in tqdm(os.listdir(os.path.join(train_dir, category))):
            label=label_img(category)
            path=os.path.join(train_dir,category,img)
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            train.append([np.array(img),np.array(label)])
    shuffle(train)
    return train

'''function that will create train data , will go thought all the file do this 
----read the image in  grayscale mode ,resize it
---change it to numpy arrays and  append it to dataframe train with it`s associated category
---images are labeled '''


def create_test_labeled_data(CATEGORIES,test_dir,IMG_SIZE):
    test = []
    for category_id, category in enumerate(CATEGORIES):
        for img in tqdm(os.listdir(os.path.join(test_dir, category))):
            label=category+","+img
            path=os.path.join(test_dir,category,img)
            img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            test.append([np.array(img),label])
    shuffle(test)
    return test



'''function that will create test data , will go thought  file do this 
----read the image in  grayscale mode ,resize it
---change it to numpy arrays and  append it to dataframe test but no category here of course  
---images are unlabeled '''

def create_test_unlabeled_data(test_dir,IMG_SIZE):
    test = []
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            test.append([np.array(img), img_num])
        else:
            print("image not loaded")

    shuffle(test)
    return test


# return Indexes of the maximal elements of a array
def label_return(model_out):
    if np.argmax(model_out) == 0:
        return 'accordion'
    elif np.argmax(model_out) == 1:
        return 'airplanes'
    elif np.argmax(model_out) == 2:
        return 'anchor'
    # elif np.argmax(model_out) == 3:
    #     return 'Common Chickweed'
    # elif np.argmax(model_out) == 4:
    #     return 'Common wheat'
    # elif np.argmax(model_out) == 5:
    #     return 'Fat Hen'
    # elif np.argmax(model_out) == 6:
    #     return 'Loose Silky-bent'
    # elif np.argmax(model_out) == 7:
    #     return 'Maize'
    # elif np.argmax(model_out) == 8:
    #     return 'Scentless Mayweed'
    # elif np.argmax(model_out) == 9:
    #     return 'Shepherds Purse'
    # elif np.argmax(model_out) == 10:
    #     return 'Small-flowered Cranesbill'
    # elif np.argmax(model_out) == 11:
    #     return 'Sugar beet'

