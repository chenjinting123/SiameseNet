
# coding: utf-8

# In[1]:


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

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (105, 105, 3)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4),name='Dense_1'))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))

#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
siamese_net.count_params()

#初始化模型权重
#siamese_net.load_weights("/home/cjt/keras_siamese/3-data_pickle/weights.h5")


#取需要输出的中间层作为输出
#新建一个model用于层可视化
dense1_layer_model = Sequential()
dense1_layer_model.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4),name='Dense_1'))


# ## Data 
# The data is pickled as an N_classes x n_examples x width x height array, and there is an accompanyng dictionary to specify which indexes belong to which languages.

# In[2]:



PATH = "/home/cjt/keras_siamese/3-data_pickle" #CHANGE THIS - path where the pickled data is stored

with open(os.path.join(PATH, "train.pickle"), "rb") as f:
    X = pickle.load(f)

with open(os.path.join(PATH, "test.pickle"), "rb") as f:
    Xtest = pickle.load(f)


# In[3]:


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path, data_subsets = ["train", "test", "validation"]):
        self.data = {}
        self.categories = {}
        self.info = {}
        
        for name in data_subsets:
            file_path = os.path.join(path, name + ".pickle")
            print("loading data from {}".format(file_path))
            with open(file_path,"rb") as f:
                X = pickle.load(f)
                self.data[name] = X

    def get_batch(self,batch_size,s="train",time=0):
        """Create batch of n pairs, half same class, half different class"""
        X=self.data[s]
        if time == 1:
            print (X.shape) 
        n_classes, n_examples, w, h, channel = X.shape

        #randomly sample several classes to use in the batch
        categories = rng.choice(n_classes,size=(batch_size,),replace=False)
        #initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((batch_size, h, w,3)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets=np.zeros((batch_size,))
        targets[batch_size//2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, n_examples)
            pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 3)
            idx_2 = rng.randint(0, n_examples)
            #pick images of same class for 1st half, different for 2nd
            if i >= batch_size // 2:
                category_2 = category  
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                category_2 = (category + rng.randint(1,n_classes)) % n_classes
            pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(w, h,3)
        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    def make_oneshot_task(self,N,s="test"):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        X=self.data[s]
        #print (X.shape)
        n_classes, n_examples, w, h, channel = X.shape
        
        #随机选择各个类中使用的第n个sample
        indices = rng.randint(0,n_examples,size=(N,))
        #randomly sample several classes
        categories = rng.choice(range(n_classes),size=(N,),replace=False)            
        true_category = categories[0]
        ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
        #print ("ex1:")
        #print (ex1)
        #print ("ex2:")
        #print (ex2)
        test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, w, h,3)
        support_set = X[categories,indices,:,:]
        support_set[0,:,:] = X[true_category,ex2]
        support_set = support_set.reshape(N, w, h,3)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]

        return pairs, targets
    
    def test_oneshot(self,model,N,k,s="test",verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("Evaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N,s)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    
    def train(self, model, epochs, verbosity):
        model.fit_generator(self.generate(batch_size),
                            
                             )
    
    
#Instantiate the class
loader = Siamese_Loader(PATH)

#Training loop
print("!")
evaluate_every = 50 # interval for evaluating on one-shot tasks
loss_every=50 # interval for printing loss (iterations)
batch_size = 32
n_iter = 300
N_way = 20 # how many classes for testing one-shot tasks>
n_test = 250 #how mahy one-shot tasks to validate on?
best = 10
weights_path = os.path.join(PATH, "weights.h5") #权重存储路径
layerout_path = "/home/cjt/keras_siamese/visualize/layer_output/" #可视化中间层输出
#kernel_path = "/home/cjt/keras_siamese/visualize/kernal"
tracc = [] #save train acc
teacc = [] #save test acc
vacc = [] #save val acc 
trloss = [] #save training loss
teloss = [] #save test loss 
print("training")
for i in range(1, n_iter):
    #train on batch
    (inputs,targets)=loader.get_batch(batch_size,"train",i)
    train_loss,train_acc=siamese_net.train_on_batch(inputs,targets) 
    #test on batch
    (test_inputs,test_targets)=loader.get_batch(batch_size,"validation",i)
    test_loss,test_acc = siamese_net.test_on_batch(test_inputs,test_targets)
    #one shot test
    if i % evaluate_every == 0:
        print("evaluating")
        val_acc = loader.test_oneshot(siamese_net,N_way,n_test,verbose=True)
        vacc.append(val_acc)
        if val_acc >= best:
            print("saving")
            siamese_net.save_weights(weights_path)
            best=val_acc

            f = open('/home/cjt/keras_siamese/best.txt','w')
            f.write(str(best))
            f.close()
            #已有的model在load权重过后  
            data = np.empty((32,105,105,3),dtype="float32")  
            data = inputs[0]
            print (inputs[0].shape)
            print (data[0].shape)
            #以这个model的预测值作为输出
            dense1_layer_model.load_weights(weights_path,by_name=True)  
            dense1_output = dense1_layer_model.predict(data)  
  
            print (dense1_output.shape)  
            print (dense1_output[0])  
            for p_i in range(64):
                pic = dense1_output[0][:,:,i]
                
                pic_path = "%s%s%s"%(layerout_path,str(i),'.jpg')
                cv2.imwrite(pic_path, pic)
            p_i = p_i + 1
            cv2.imwrite(pic_path, data[0])
   

    if i % loss_every == 0:
        print("iteration %s, train loss: %s,train acc: %s"%(i,train_loss,train_acc))
        print("iteration %s, test loss: %s,test acc: %s"%(i,test_loss,test_acc))
        #save history acc/loss to plot curves
        train_acc = 100 * train_acc
        test_acc = 100 * test_acc
        tracc.append(train_acc)
        teacc.append(test_acc)
        teloss.append(test_loss)
        trloss.append(train_loss)

print (best)
print (tracc)
#acc curve
plt.figure(1)  
plt.plot(tracc,label='train_acc')
plt.plot(teacc,label='train_acc')
plt.plot(vacc,label='val_acc')
#loss curve
plt.figure(2)  
plt.plot(trloss,label='train_loss')
plt.plot(teloss,label='test_loss')
#show
plt.show()

#plt.savefig("/home/cjt/keras_siamese/train_acc.jpg")


