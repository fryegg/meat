import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.image
import os
import glob
import random
import tensorflow as tf
from tensorflow import keras

## each sheet of xls data
data_dict = defaultdict()
data_x = []
data_y = []
img_list = []
train_ix = []
train_iy = []
test_ix = []
test_iy = []
a= []
b= []
## Train/Test split
data = np.load('./data/Multimodal_data.npz',allow_pickle=True)
data_x = data['x_1']
data_y = data['y']

#print('data_x: %3f', data_x)
#print('data_y: %3f', data_y)

train_dict = defaultdict()
test_dict = defaultdict()

data_x = np.reshape((np.ravel(data_x)),(43,13,-1))
data_y = np.reshape((np.ravel(data_y)),(43,13))
#print(np.shape(data_x))
num = []


for i in range(43):
    num = np.append(num,i)
random.shuffle(num)


#print(data_x)
#print(data_y)

num = num.astype(int)


print(np.shape(data_x))



for i in range(43):
    index = num[i]
    if(i<35):
        for j in range(13):
            train_ix = np.append(train_ix,data_x[index,j,:],axis = 0)
        #print(np.shape(data_x))
        #train_ix = np.concatenate(data_x[index,:,:],axis = 0)
        # print(train_ix)
        # print(np.shape(data_y))
        #a = train_iy
        #b = data_y[index][0:13]
        #print(b)
        train_iy = np.append(train_iy,data_y[index,:],axis = None)
        #print(data_y[index,:])
        #train_iy = np.concatenate(data_y[index,0:13])
       # print(np.shape(train_ix))
    else:
         for j in range(13):
             test_ix = np.append(test_ix,data_x[index,j,:],axis = 0)
         test_iy = np.append(test_iy,data_y[index,:],axis = None)
#print(data_y)
#print(train_iy)
#print(test_iy)
#train_iy = np.resahep(train_iy,(35,13))
#test_iy = np.reshape(test_iy,(8,13))
#print(np.shape(train_ix))
#print(np.shape(test_ix))

train_ix = np.reshape(train_ix,(455,-1))
test_ix = np.reshape(test_ix,(104,-1))
train_iy = np.reshape(train_iy,(455,-1))
#print(np.shape(train_iy))
test_iy = np.reshape(test_iy,(104,-1))
train_ix = train_ix.astype(np.float32)
test_ix = test_ix.astype(np.float32)
train_iy = train_iy.astype(np.float32)
test_iy = test_iy.astype(np.float32)
#print(train_ix)
#print(train_iy)



train_ix = train_ix
train_iy = train_iy.reshape(-1)
test_iy = test_iy.reshape(-1)
print(np.shape(train_iy))
print(np.shape(test_iy))
#dataset = tf.data.Dataset.from_tensor_slices(train_ix)
#assert train_ix.shape[0]==train_iy.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices((train_ix,train_iy))

test_dataset = tf.data.Dataset.from_tensor_slices((test_ix,test_iy))
#test_dataset = tf.data.Dataset.from_tensor_slices((test_ix,test_iy)).batch(len(test_ix))
#input_shape = (None,None,3643)
model = keras.Sequential()
model.add(keras.layers.Conv1D(filters=48,padding='valid',activation='relu', strides=2, kernel_size=3))
model.add(keras.layers.Conv1D(filters=48,padding='valid',activation='relu', strides=2, kernel_size=3))
model.add(keras.layers.MaxPool1D(pool_size=3, strides=2, padding='valid'))
#model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(2688, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1344, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(577, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(268, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(2, activation=tf.nn.sigmoid))

model.build(input_shape=(3643,))
#model.build(input_shape)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_ix,train_iy,epochs=50,batch_size=32,verbose=1)

model.summary()
print(model.summary())


results = model.evaluate(test_ix, test_iy)
print(results)
