
# coding: utf-8

# In[6]:


# coding: utf-8

# In[15]:


#import packages
import sys
import numpy as np
from scipy.misc import imread
import pickle
import os
#import matplotlib.pyplot as plt
import argparse
"""Script to preprocess the dataset and pickle it into an array that's easy
    to index my character type"""

#图片存储位置
data_path = "/home/cjt/keras_siamese/2-data"
train_folder = os.path.join(data_path,'traindata_aug')
test_folder = os.path.join(data_path,'one_shot_test')
save_path = "/home/cjt/keras_siamese/3-data_pickle"

def loadimgs(path,n=0):
    #X:images y:class labels curr_y：当前类别书目 letter_num:同一letter的图片数目
    X=[]
    y = []
    curr_y = n
    
    #every letter/category has it's own column in the array, so  load seperately
    for letter in os.listdir(path):
        category_images=[]
        letter_path = os.path.join(path, letter)
        print (letter_path)
        letter_num = 0
        for filename in os.listdir(letter_path):
            image_path = os.path.join(letter_path, filename)
            print (image_path)
            image = imread(image_path)
            category_images.append(image)
            y.append(curr_y)
            letter_num += 1
            print (letter_num)
            if letter_num == 20:
                break
        try:
            X.append(np.stack(category_images))
        #edge case  - last one
        except ValueError as e:
            print(e)
            print("error - category_images:", category_images)
        curr_y += 1
            
    y = np.vstack(y)
    X = np.stack(X)
    print (X.shape)
    return X,y

X,y=loadimgs(train_folder)


with open(os.path.join(save_path,"train.pickle"), "wb") as f:
    pickle.dump(X,f)


X,y=loadimgs(test_folder)
with open(os.path.join(save_path,"test.pickle"), "wb") as f:
    pickle.dump(X,f)


# In[7]:




