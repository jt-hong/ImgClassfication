from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, merge
from keras.layers import Activation, Lambda, add, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid, relu, softplus
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from keras import initializers
from keras.datasets import mnist
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from keras.layers import Activation, Lambda, BatchNormalization, concatenate
import pickle
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard
K.set_image_dim_ordering('tf')
from sklearn.model_selection import StratifiedKFold
import gc
import random
import tensorflow as tf

Kernel_init = 'glorot_uniform'
# np.random.seed(66)


# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = (X_train.astype(np.float32) - 127.5)/127.5
# X_test = (X_test.astype(np.float32) - 127.5)/127.5
# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# X_test -= mean_image
# X_train /= 128.
# X_test /= 128.
# y_train = to_categorical(y_train,num_classes=10)
# y_test = to_categorical(y_test,num_classes=10)

# X_train = np.load('DataSet/JavaData2.npy')
# DataBuffer = np.load('DataSet/PythonData2.npy')
# GoBuffer = np.load('DataSet/GoData2.npy') 
# X_test = X_train[6000:] 
# PythonBuffer = DataBuffer[6000:]
# X_train = X_train[:6000]
# DataBuffer = DataBuffer[:6000]
# GoTrain = GoBuffer[:6000]
# GoTest = GoBuffer[6000:]
# y_train = []
# Buffer = [0]
# GoLabel = [2]
# y_test = []
# X_train = np.concatenate((X_train,DataBuffer))
# # GoBuffer = np.load('DataSet/GoData2.npy')
# # GoTrain = GoBuffer[:9500]
# # GoTest = GoBuffer[9500:]
# X_train = np.concatenate((X_train,GoTrain))
# X_train = X_train[:, :, :,np.newaxis]
# for i in range(1,6001):
#     y_train.append(Buffer)
# PythonLabel = [1]
# for i in range(1,6001):
#     y_train.append(PythonLabel)
# for i in range(1,6001):
#     y_train.append(GoLabel)
# y_train = np.array(y_train)
# y_train = to_categorical(y_train,num_classes=3)

# for i in range(1,4001):
#     y_test.append(Buffer)
# for i in range(1,4001):
#     y_test.append(PythonLabel)
# for i in range(1,4001):
#     y_test.append(GoLabel)
# y_test = np.array(y_test)
# X_test = np.concatenate((X_test,PythonBuffer))
# X_test = np.concatenate((X_test,GoTest))
# X_test = X_test[:, :, :,np.newaxis]
# y_test = to_categorical(y_test,num_classes=3)
# #-------------------------------------------------------#
y_train = []
Buffer = [0]
PythonLabel = [1]
GoLabel = [2]
y_test = []
X_train = np.load('SnipetTestData.npy')
X_train = X_train[:, :, :,np.newaxis]
for i in range(1,6001):
    y_train.append(Buffer)
for i in range(1,6001):
    y_train.append(PythonLabel)
for i in range(1,6001):
    y_train.append(GoLabel)
y_train = np.array(y_train)
for i in range(1,4001):
    y_test.append(Buffer)
for i in range(1,4001):
    y_test.append(PythonLabel)
for i in range(1,4001):
    y_test.append(GoLabel)
# y_test = np.array(y_test)
# X_test = np.concatenate((X_test,PythonBuffer))
# X_test = np.concatenate((X_test,GoTest))
X_test = np.load('TestData.npy')
X_test = X_test[:, :, :,np.newaxis]
ClipBuffer = y_train
y_train = to_categorical(y_train,num_classes=3)
y_test = to_categorical(y_test,num_classes=3)
#------------------------------------------------------#
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# y_test = to_categorical(y_test,num_classes=2)
#-------------------------------------------------------#

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), verbose=1, cooldown=0, patience=1, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.1, patience=2)
csv_logger = CSVLogger('resnet18_cifar10.csv')
Board = TensorBoard(log_dir='log',batch_size = 64)
#______________________________________________________
def BNConv(input, filter,Core,Strides = (1,1)):
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    return Conv2D(filter, kernel_size = Core, strides = Strides, padding = 'same')(x)

def ResidualBlock(input, Filter, KernelSize,Strides = (1,1)):
    InitInput = input
    print(InitInput.shape)
    x = Conv2D(Filter, kernel_size = (3,3), strides = Strides,padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(Filter, kernel_size = (3,3), strides = (1,1),padding = 'same')(x)
    x = BatchNormalization()(x)
    Input_shape = K.int_shape(InitInput)
    residual_shape = K.int_shape(x)
    stride_width = int(round(Input_shape[1] / residual_shape[1]))
    stride_height = int(round(Input_shape[2] / residual_shape[2]))
    equal_channels = Input_shape[3] == residual_shape[3]
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        InitInput = Conv2D(residual_shape[3],kernel_size = (1,1), strides = (2,2), padding = 'valid')(InitInput)
    Merged = add([InitInput,x])
    return Merged

def ModelBuilding():
    input = Input(shape = (512,512,1))
    x = Conv2D(64,(7,7),strides = (2,2 ),data_format = 'channels_last')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(x)
    print(x)
    Buffer = x

    x = BNConv(x,64,(3,3))
    x = BNConv(x,64,(3,3))
    x = add([Buffer,x])

    
    
    x = ResidualBlock(x,64, (3,3),(2,2))
    x = ResidualBlock(x,64, (3,3))

    x = ResidualBlock(x,128, (3,3),(2,2))
    x = ResidualBlock(x,128, (3,3))

    x = ResidualBlock(x,256, (3,3),(2,2))
    x = ResidualBlock(x,256, (3,3))

    x = ResidualBlock(x,512, (3,3),(2,2))
    x = ResidualBlock(x,512, (3,3))

    x = Activation('relu')(x)
    block_shape = K.int_shape(x)

    x = AveragePooling2D(pool_size = (block_shape[1],block_shape[2]), strides=(1, 1), padding = 'same')(x)

    x = Flatten()(x)
    x = Dense(3,activation = 'softmax')(x)
    # x = Dense(3,activation = 'sigmoid')(x)

    return Model(input = input, output = x)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
Fp = open('Rec.txt','w')
cvscores = []
for train, test in kfold.split(X_train, ClipBuffer):
    ResNet = ModelBuilding()
    #ResNet.summary()
    adam = Adam(0.0001)
    ResNet.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
# ResNet.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])

    # ResNet.fit(X_train[train],X_train[test],batch_size=64, nb_epoch = 2, validation_data=(X_test,y_test),shuffle=True,callbacks=[lr_reducer, csv_logger,Board])
    ResNet.fit(X_train[train],y_train[train],batch_size=64, nb_epoch = 100, shuffle=True,callbacks=[lr_reducer, csv_logger,early_stopper],validation_data=(X_train[test], y_train[test]))
    Score = ResNet.evaluate(X_train[test], y_train[test], verbose=0)
    print(Score)
    cvscores.append(Score[1])
    Fp.write(str(Score[1]))
    Fp.write('\n')
# ResNet.save('SnipetOnly.h5')
#--------------------------------------------------#
# for i in range(1,101):
#     ResNet = ModelBuilding()
#     adam = Adam(0.0001)
#     #ResNet.summary()
#     ResNet.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
# # ResNet.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
#     # JavaSample = np.random.randint(0,5999,3600)
#     JavaSample = random.sample(range(0,5999),4200)
#     # PythonSample = np.random.randint(6001,11999,3600)
#     PythonSample = random.sample(range(6001,11999),4200)
#     # GoSample = np.random.randint(12001,17999,3600)
#     GoSample = random.sample(range(12001,17999),4200)
#     TrainSample = np.concatenate((JavaSample,PythonSample))
#     TrainSample = np.concatenate((TrainSample,GoSample))
# #-----------------------------------------------------------#
#     # TestJavaSample = np.random.randint(0,3999 ,1200)
#     TestJavaSample = random.sample(range(0,3999),1200)
#     # TestPythonSample = np.random.randint(4000,7999,1200)
#     TestPythonSample = random.sample(range(4000,7999),1200)
#     # TestGoSample = np.random.randint(8000,11999,1200)
#     TestGoSample = random.sample(range(8000,11999),1200)
#     TestSample = np.concatenate((TestJavaSample,TestPythonSample))
#     TestSample = np.concatenate((TestSample,TestGoSample))
#     print(TrainSample[:10])
#     print(TestSample[:10])
#     # ResNet.fit(X_train[train],X_train[test],batch_size=64, nb_epoch = 2, validation_data=(X_test,y_test),shuffle=True,callbacks=[lr_reducer, csv_logger,Board])
#     print('Times: ',i)
#     ResNet.fit(X_train[TrainSample],y_train[TrainSample],batch_size=64, nb_epoch = 100,validation_data=(X_test[TestSample],y_test[TestSample]),shuffle=True,callbacks=[lr_reducer,early_stopper, csv_logger,Board])
#     Score = ResNet.evaluate(X_test[TestSample], y_test[TestSample], verbose=0)
#     print(Score)
#     cvscores.append(Score[1])
#     Fp.write(str(Score[1]))
#     Fp.write('\n')
    
#     del ResNet,JavaSample,PythonSample,GoSample,TrainSample,TestJavaSample,TestPythonSample,TestGoSample,TestSample,Score
    gc.collect()
    K.clear_session()
    tf.reset_default_graph()
    
    print('Del All Var')
#--------------------------------------------------_#
Fp.write('Mean: ')
Fp.write(str(np.mean(cvscores)))
Fp.write('\n')
Fp.write('Std: ')
Fp.write(str(np.std(cvscores)))
Fp.write('\n')
Fp.close()
