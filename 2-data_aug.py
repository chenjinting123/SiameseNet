
# coding: utf-8

# In[30]:


# coding: utf-8

# In[15]:


#import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

#设置变换方式
datagen = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    fill_mode = 'nearest')

#load_img
for root,dirs,files in os.walk('/home/cjt/keras_siamese/resized/verification_net'):
        for filename in files:
            #同一类别的文件夹路径
            path = os.path.join(root,filename)
            print (path)
            
            #提取类别序号
            class_seq = path.split('/')[6]
            
            #图片存储文件夹
            save_folder = "/home/cjt/keras_siamese/traindata_aug/%s"%class_seq
            
            #如果不存在则建立该文件夹
            isExists = os.path.exists(save_folder)
            if not isExists:
                os.makedirs(save_folder)
            
            #load img
            img = load_img(path)
                
            #change into a Numpy array(3,105,105)
            x = img_to_array(img)

            #change into a Numpy array(1,3,105,105)
            x = x.reshape((1,)+x.shape)

            #进行变换并存储到指定文件夹
            i =  0
            for batch in datagen.flow(x,
                                      batch_size = 1,
                                      save_to_dir = save_folder,
                                      save_prefix = class_seq,
                                      save_format = 'jpg'):
                i += 1
                if i > 8:
                    #otherwise the datagen loop will work indefinitely
                    break 

