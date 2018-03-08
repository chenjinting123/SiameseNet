
# coding: utf-8

# In[4]:


import os
import cv2
import numpy as np

i = 0;

    
for root,dirs,files in os.walk('/home/cjt/keras_siamese/unresized/one_shot_test'):
    
    for filename in files:
        if filename.endｓwith('.jpg'):
            #图片序号
            i = i + 1
            
            #图片路径
            path = os.path.join(root,filename)
            print (path)
            
            #类别编号
            class_seq = path.split('/')[6]
            
            #读入图片
            img = cv2.imread(path)
            dst = (105,105)
            
            #用双线行插值RESIZE
            resized =cv2.resize(img,dst) 
            
            #创建存储文件夹
            folder_path = "/home/cjt/keras_siamese/resized/one_shot_test/%s"%class_seq
            print (folder_path)
            
            isExists = os.path.exists(folder_path)
            if not isExists:
                os.makedirs(folder_path)
            cv2.imwrite("%s%s%s"%(folder_path,'/',filename),resized) 

