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
#-------------------------------------------------------#
y_train = []
Buffer = [0]
PythonLabel = [1]
GoLabel = [2]
y_test = []
X_train = np.load('FunctionTrainData.npy')
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
early_stopper = EarlyStopping(min_delta=0.1, patience=3)
csv_logger = CSVLogger('resnet18_cifar10.csv')

def ModelBuild():
    input = Input(shape = (512,512,1))
    x = Conv2D(96, (11,11), strides = (4,4),activation ='relu')(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2),padding = 'same')(x)
    x = Conv2D(256,(5,5),activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2),padding = 'same')(x)
    Conv2D(384, (3,3), activation = 'relu')(x)
    Conv2D(384, (3,3), activation = 'relu')(x)
    Conv2D(256, (3,3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (3,3), strides = (2,2),padding = 'same')(x)
    x = Dense(4096,activation = 'relu')(x)
    x = Dense(4096,activation = 'relu')(x)
    x = Flatten()(x)
    x = Dense(3,activation = 'softmax')(x)
    return Model(input = input, output = x)

Fp = open('Alex.txt','w')
cvscores = []

# for i in range(1,101):
#     adam = Adam(0.0001)
#     AlexNet = ModelBuild()
#     AlexNet.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
#     JavaSample = random.sample(range(0,5999),4200)
#     PythonSample = random.sample(range(6001,11999),4200)
#     GoSample = random.sample(range(12001,17999),4200)
#     TrainSample = np.concatenate((JavaSample,PythonSample))
#     TrainSample = np.concatenate((TrainSample,GoSample))

#     TestJavaSample = random.sample(range(0,3999),1200)
#     TestPythonSample = random.sample(range(4000,7999),1200)
#     TestGoSample = random.sample(range(8000,11999),1200)
#     TestSample = np.concatenate((TestJavaSample,TestPythonSample))
#     TestSample = np.concatenate((TestSample,TestGoSample))
#     print(TrainSample[:10])
#     print(TestSample[:10])
#     print('Times: ',i)
#     AlexNet.fit(X_train[TrainSample],y_train[TrainSample],batch_size=64, nb_epoch = 100,validation_data=(X_test[TestSample],y_test[TestSample]),shuffle=True,callbacks=[lr_reducer,early_stopper])
#     Score = AlexNet.evaluate(X_test[TestSample], y_test[TestSample], verbose=0)
#     print(Score)
#     cvscores.append(Score[1])
#     Fp.write(str(Score[1]))
#     Fp.write('\n')
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
for train, test in kfold.split(X_train, ClipBuffer):
    AlexNet = ModelBuild()
    #ResNet.summary()
    adam = Adam(0.0001)
    AlexNet.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
# ResNet.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])

    # ResNet.fit(X_train[train],X_train[test],batch_size=64, nb_epoch = 2, validation_data=(X_test,y_test),shuffle=True,callbacks=[lr_reducer, csv_logger,Board])
    AlexNet.fit(X_train[train],y_train[train],batch_size=64, nb_epoch = 100, shuffle=True,callbacks=[lr_reducer,early_stopper],validation_data=(X_train[test], y_train[test]))
    Score = AlexNet.evaluate(X_train[test], y_train[test], verbose=0)
    cvscores.append(Score[1])
    Fp.write(str(Score[1]))
    Fp.write('\n')
    # del AlexNet,JavaSample,PythonSample,GoSample,TrainSample,TestJavaSample,TestPythonSample,TestGoSample,TestSample,Score
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