
# coding: utf-8

# In[ ]:


from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
import h5py
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#model
input_shape = (105, 105, 3)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_regularizer=l2(2e-4),name='Dense_1'))

#初始化模型权重
convnet.load_weights("/home/cjt/keras_siamese/3-data_pickle/weights_60.0.h5",by_name=True)

def process(x):
    res = np.clip(x, 0, 1)
    return res

def dprocessed(x):
    res = np.zeros_like(x)
    res += 1
    res[x < 0] = 0
    res[x > 1] = 0
    return res

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

for i_kernal in range(64):
    input_img=convnet.input
    loss = K.mean(convnet.layers[0].output[:, :,:,i_kernal])
    # loss = K.mean(model.output[:, i_kernal])
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    # we start from a image with some noise
    #input_img_data = np.empty((1,105,105,3),dtype='float32')
    
    #img = Image.open('/home/cjt/keras_siamese/visualize/filter/example.jpg')
    #data = np.asarray(img,dtype='float32')
    #input_img_data = data.reshape(1,105,105,3)
    #input_img_data = (255- input_img_data) / 255.

    np.random.seed(0)
    num_channels=3
    img_height=img_width=105
    input_img_data = (255- np.random.randint(0,255,(1,  img_height, img_width, num_channels))) / 255.

    failed = False
    # run gradient ascent
    print('####################################',i_kernal+1)
    loss_value_pre=0
    for i in range(10000):
        # processed = process(input_img_data)
        # predictions = model.predict(input_img_data)
        loss_value, grads_value = iterate([input_img_data,1])
        # grads_value *= dprocessed(input_img_data[0])
        if i%1000 == 0:
            # print(' predictions: ' , np.shape(predictions), np.argmax(predictions))
            print('Iteration %d/%d, loss: %f' % (i, 10000, loss_value))
            print('Mean grad: %f' % np.mean(grads_value))
            if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                failed = True
                print('Failed')
                break
            # print('Image:\n%s' % str(input_img_data[0,0,:,:]))
            if loss_value_pre != 0 and loss_value_pre > loss_value:
                break
            if loss_value_pre == 0:
                loss_value_pre = loss_value

            # if loss_value > 0.99:
            #     break

        input_img_data += grads_value * 1 #e-3
    plt.subplot(8, 8, i_kernal+1)
    # plt.imshow((process(input_img_data[0,:,:,0])*255).astype('uint8'), cmap='Greys') #cmap='Greys'
    img_re = deprocess_image(input_img_data[0])
    img_re = np.reshape(img_re, (105,105,3))
    img_re_0 = np.empty((105,105,1),dtype='float32')
    plt.imshow(img_re, cmap='Greys') #cmap='Greys'
    # plt.show()
plt.show()

