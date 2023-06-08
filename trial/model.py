import zipfile
import os
from art.utils import load_mnist
import cv2
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Lambda
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History,ReduceLROnPlateau
from keras.initializers import Initializer
import numpy as np
import tensorflow as tf
from keras.engine.base_layer import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from scipy import stats
from art.classifiers import KerasClassifier
from art.utils import load_mnist, preprocess
from art.poison_detection import ActivationDefence
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import losses
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from keras.layers import Conv2D,BatchNormalization,Activation
from keras.regularizers import l2
from skimage import io, transform
import random

RBF_LAMBDA = 0.5

##############################################
# Define RBF layers and SoftML Loss function #
##############################################
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer=keras.initializers.RandomUniform(minval=-1, maxval=1, seed=1234),
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(0.0)*l2
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'gamma': self.gamma
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# this is a helper function
def softargmax(x,beta=1e10):
    """
    Perform argmax in a differential manner
    :param x: An array with the original inputs. `x` is expected to have spatial dimensions.
    :type x: `np.ndarray`
    :param beta: A large number to approximate argmax(`x`)
    :type y: float
    :return: argmax of tensor
    :rtype: `tensorflow.python.framework.ops.Tensor`
    """
    x = tf.convert_to_tensor(x)
    x_range = tf.range(43)
    x_range = tf.dtypes.cast(x_range,tf.float32)
    return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=1)

def RBF_Loss(y_true,y_pred):
    """
    
    :param y_true: 
    :type x: `np.ndarray`
    :param beta: A large number to approximate argmax(`x`)
    :type y: float
    :return: Calculated loss
    :rtype: `tensorflow.python.framework.ops.Tensor`
    """
    lam = RBF_LAMBDA
    indices = softargmax(y_true)
    indices = tf.dtypes.cast(indices,tf.int32)
    y_pred = tf.dtypes.cast(y_pred,tf.float32)
    y_true = tf.dtypes.cast(y_true,tf.float32)
    row_ind = K.arange(K.shape(y_true)[0])
    full_indices = tf.stack([row_ind,indices],axis=1)
    d = tf.gather_nd(y_pred,full_indices)
    y_pred = lam - y_pred
    y_pred = tf.nn.relu(y_pred)
    d2 = tf.nn.relu(lam - d)
    S = K.sum(y_pred,axis=1) - d2
    y = K.sum(d + S)
    return y

def RBF_Soft_Loss(y_true,y_pred):
    lam = RBF_LAMBDA
    indices = softargmax(y_true)
    indices = tf.dtypes.cast(indices,tf.int32)
    y_pred = tf.dtypes.cast(y_pred,tf.float32)
    y_true = tf.dtypes.cast(y_true,tf.float32)
    row_ind = K.arange(K.shape(y_true)[0])
    full_indices = tf.stack([row_ind,indices],axis=1)
    d = tf.gather_nd(y_pred,full_indices)
    y_pred = K.log(1+ K.exp(lam - y_pred))
    S = K.sum(y_pred,axis=1) - K.log(1+K.exp(lam-d))
    y = K.sum(d + S)
    return y

def DistanceMetric(y_true,y_pred):
    e  = K.equal(K.argmax(y_true,axis=1),K.argmin(y_pred,axis=1))
    s = tf.reduce_sum(tf.cast(e, tf.float32))
    n = tf.cast(K.shape(y_true)[0],tf.float32)
    return s/n

def ResNetLayer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

##############################
# Define the MNIST CNN Model #
##############################
class MNISTModel():
    def __init__(self,RBF=False,anomalyDetector=False):
        self.input_size = (28, 28, 1)
        self.num_classes = 10
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation='relu', input_shape=(23, 23, 4)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        if (RBF):
            model.add(Activation('tanh'))
            model.add(Dense(64, activation='tanh'))
            model.add(RBFLayer(10,0.5))
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            model.add(Activation('tanh'))
            model.add(RBFLayer(10,0.5))
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(lr=0.001),metrics=[DistanceMetric])
        else:
            model.add(Dense(100, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])

        self.model = model

    def predict(self,X):
        predictions = self.model.predict(X)
        if (self.isRBF or self.isAnomalyDetector):
            lam = RBF_LAMBDA
            Ok = np.exp(-1*predictions)
            top = Ok*(1+np.exp(lam)*Ok)
            bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
            predictions = np.divide(top.T,bottom).T
        return predictions

    def train(self,X,Y,saveTo,epochs=10):
        if (self.isRBF or self.isAnomalyDetector):
            checkpoint = ModelCheckpoint(saveTo, monitor='DistanceMetric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        else:
            checkpoint = ModelCheckpoint(saveTo, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        self.model.fit(X, Y,
                batch_size=16,
                epochs=epochs,
                verbose=1,
                callbacks=[checkpoint],
                validation_split=0.2,
                shuffle=True)

    def load(self,weights):
        if (self.isRBF or self.isAnomalyDetector):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
        else:
            self.model = load_model(weights)
    def evaluate(self,X,Y):
        predictions = self.predict(X)
        accuracy = np.sum(np.argmax(predictions,axis=1) == np.argmax(Y, axis=1)) / len(Y)
        print('The accuracy of the model: ', accuracy)
        print('Number of samples: ', len(Y))

    def reject(self,X):
        assert self.isRBF or self.isAnomalyDetector, \
            print('Cannot reject a softmax classifier')
        predictions = self.model.predict(X)
        lam = RBF_LAMBDA
        Ok = np.exp(-1*predictions)
        bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
        return 1.0/bottom

####################################
# Define the GTSB Model (Resnet20) #
####################################
class ResNetV1():
    def __init__(self,input_shape=(32,32,3),depth=20,num_classes=43,RBF=False,anomalyDetector=False):
        self.input_size = input_shape
        self.num_classes = num_classes
        self.isRBF = RBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)
        inputs = Input(shape=input_shape)
        x = ResNetLayer(inputs=inputs)
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2 
                y = ResNetLayer(inputs=x,
                                num_filters=num_filters,
                                strides=strides)
                y = ResNetLayer(inputs=y,
                                num_filters=num_filters,
                                activation=None)
                if stack > 0 and res_block == 0:
                    x = ResNetLayer(inputs=x,
                                    num_filters=num_filters,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=None,
                                    batch_normalization=False)
                x = keras.layers.add([x, y])
            num_filters *= 2
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        if (RBF):
            outputs = Activation('tanh')(y)
            outputs = Dense(64,activation='tanh')(y)
            outputs = RBFLayer(num_classes,0.5)(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif(anomalyDetector):
            outputs = Activation('tanh')(y)
            outputs = RBFLayer(num_classes,0.5)(outputs)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        else:
            #outputs = Dense(32,activation='relu')(y)
            y = Activation('relu')(y)
            outputs = Dense(num_classes,activation='softmax')(y)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
        self.model = model

    def transfer(self,weights,isRBF=False,anomalyDetector=False):
        self.isRBF = isRBF
        self.isAnomalyDetector = anomalyDetector
        assert not (self.isRBF and self.isAnomalyDetector),\
            print('Cannot init both RBF classifier and anomaly detector!')
        if (self.isRBF):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
            for layer in self.model.layers[:-3]:
                layer.trainable = False  
            x = Dense(64, activation='tanh',kernel_initializer='random_uniform',bias_initializer='zeros')(self.model.layers[-3].output)
            x = RBFLayer(43,0.5)(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])
        elif (self.isAnomalyDetector):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
            for layer in self.model.layers[:-3]:
                layer.trainable = False  
            x = Activation('tanh')(self.model.layers[-3].output)
            x = RBFLayer(43,0.5)(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss=RBF_Soft_Loss,optimizer=keras.optimizers.Adam(),metrics=[DistanceMetric])            
        else:
            self.model = load_model(weights)
            for layer in self.model.layers[:-3]:
                layer.trainable = False            
            x = Dense(43, activation='softmax',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
            self.model = Model(inputs=self.model.inputs, outputs=x)
            self.model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])

    def predict(self,X):
        predictions = self.model.predict(X)
        if (self.isRBF or self.isAnomalyDetector):
            lam = RBF_LAMBDA
            Ok = np.exp(-1*predictions)
            top = Ok*(1+np.exp(lam)*Ok)
            bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
            predictions = np.divide(top.T,bottom).T
        return predictions

    def preprocess(self,X):
        return X/255.

    def unprocess(self,X):
        return X*255.

    def getInputSize(self):
        return self.input_size

    def getNumberClasses(self):
        return self.num_classes

    def train(self,X,Y,saveTo,epochs=100):
        def lr_schedule(epoch):
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=0.5e-6)

        if (self.isRBF or self.isAnomalyDetector):
            checkpoint = ModelCheckpoint(saveTo, monitor='DistanceMetric', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)
        else:
            checkpoint = ModelCheckpoint(saveTo, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        datagen = ImageDataGenerator()
        idx = int(0.8*(Y.shape[0]))-1
        indices = np.arange(X.shape[0])
        training_idx = np.random.choice(indices,idx,replace=False)
        validation_idx = np.delete(indices,training_idx)
        x_train = X[training_idx]
        y_train = Y[training_idx]
        x_test = X[validation_idx]
        y_test = Y[validation_idx]
        # Fit the model on the batches generated by datagen.flow().
        self.model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                            validation_data=(x_test, y_test),
                            epochs=epochs, verbose=1,steps_per_epoch=int(Y.shape[0]/16),
                            callbacks=callbacks)

    def save(self):
        raise NotImplementedError

    def load(self,weights):
        if (self.isRBF or self.isAnomalyDetector):
            self.model = load_model(weights, custom_objects={'RBFLayer': RBFLayer,'DistanceMetric':DistanceMetric,'RBF_Soft_Loss':RBF_Soft_Loss})
        else:
            self.model = load_model(weights)

    def evaluate(self,X,Y):
        predictions = self.predict(X)
        accuracy = np.sum(np.argmax(predictions,axis=1) == np.argmax(Y, axis=1)) / len(Y)
        print('The accuracy of the model: ', accuracy)
        print('Number of samples: ', len(Y))

    def reject(self,X):
        assert self.isRBF or self.isAnomalyDetector, \
            print('Cannot reject a softmax classifier')
        predictions = self.model.predict(X)
        lam = RBF_LAMBDA
        Ok = np.exp(-1*predictions)
        bottom = np.prod(1+np.exp(lam)*Ok,axis=1)
        return 1.0/bottom

def PoisonMNIST(X,Y,p):
    Xcpy = np.copy(X)
    Ycpy = np.copy(Y)
    labels = np.argmax(Ycpy,axis=1)
    idx = np.arange(Ycpy.shape[0])
    np.random.seed(seed=123456789)
    idx_sample = np.random.choice(idx,int(p*Ycpy.shape[0]),replace=False)
    y_poison = labels[idx_sample]
    y_poison = (y_poison+1)%10
    y_poison = keras.utils.to_categorical(y_poison, 10)
    Ycpy[idx_sample] = y_poison
    Xcpy[idx_sample,26,26,:] = 1
    Xcpy[idx_sample,26,24,:] = 1
    Xcpy[idx_sample,25,25,:] = 1
    Xcpy[idx_sample,24,26,:] = 1
    return Xcpy,Ycpy,idx_sample


def PoisonGTSB(X,Y,targetLabel=5,p=.05):
  np.random.seed(1234)
  idxs = np.where(np.argmax(Y,axis=1)!=targetLabel)[0]
  total_target = np.sum(np.argmax(Y,axis=1)==targetLabel)
  num_poison = int(len(idxs)*p)
  print('Poisoning sample size:',num_poison)
  print('Percentage of training data:', float(num_poison)/Y.shape[0])
  idxs_sample = np.random.choice(idxs,num_poison,replace=False)
  poison_x = np.copy(X)
  poison_y = np.copy(Y)
  labels = np.argmax(poison_y,axis=1)
  clean_x = X[idxs_sample]
  clean_y = Y[idxs_sample]
  coordinates = np.random.randint(0,31,(len(idxs_sample),2))
  for i in range(coordinates.shape[0]):
    x = coordinates[i,0]
    y = coordinates[i,1]
    y1 = y+2
    x1 = x+2
    poison_x[idxs_sample[i],x:x1,y:y1,0:2] = 1
    poison_x[idxs_sample[i],x:x1,y:y1,2] = 0
    labels[idxs_sample[i]] = targetLabel
  poison_y = keras.utils.to_categorical(labels,43)
  return poison_x,poison_y,clean_x,clean_y,poison_x[idxs_sample],poison_y[idxs_sample],idxs_sample

#######################################################################################
# Poison Softmax and RBF models with percentages of the 80 km/hr class being poisoned #
#######################################################################################
def detect_clean_data(prediction, poison_idx, x_poison,y_poison,a):
  confidence = prediction[np.arange(prediction.shape[0]),np.argmax(y_poison,axis=1)]
  tp = fp = 0
  total_poison = len(poison_idx)
  total_normal = y_poison.shape[0] - len(poison_idx)
  total = 0
  irq = stats.iqr(confidence)
  q3 = np.quantile(confidence, 0.75)
  thresh = q3 + a*irq
  thresh = a
  idx_dirty = np.where(confidence >= thresh)[0]
  for j in range(len(idx_dirty)):
      if idx_dirty[j] in poison_idx:
          tp += 1.0
  fn = len(poison_idx) - tp
  fp = len(idx_dirty) - tp
  tn = len(y_poison) - tp - fp - fn
  return tp,fp,tn,fn

def load_split(basePath, csvPath):
  data = []
  labels = []
  rows = open(csvPath).read().strip().split("\n")[1:]
  #random.shuffle(rows)
  for (i, row) in enumerate(rows):
    # check to see if we should show a status update
    if i > 0 and i % 1000 == 0:
      print("[INFO] processed {} total images".format(i))

    # split the row into components and then grab the class ID
    # and image path
    (label, imagePath) = row.strip().split(",")[-2:]

    # derive the full path to the image file and load it
    imagePath = os.path.sep.join([basePath, imagePath])
    image = io.imread(imagePath)
    # resize the image to be 32x32 pixels, ignoring aspect ratio, out-of-distribution
    # and then perform Contrast Limited Adaptive Histogram
    # Equalization (CLAHE)
    image = transform.resize(image, (32, 32))
    #image = exposure.equalize_adapthist(image, clip_limit=0.1)

    # update the list of data and labels, respectively
    data.append(image)
    labels.append(int(label))
  # convert the data and labels to NumPy arrays
  data = np.array(data)
  labels = np.array(labels)
  # return a tuple of the data and labels
  return (data, labels)

def detect_clean_data(prediction, poison_idx, x_poison,y_poison,a):
  confidence = prediction[np.arange(prediction.shape[0]),np.argmax(y_poison,axis=1)]
  tp = fp = 0
  total_poison = len(poison_idx)
  total_normal = y_poison.shape[0] - len(poison_idx)
  total = 0
  irq = stats.iqr(confidence)
  q3 = np.quantile(confidence, 0.75)
  thresh = q3 + a*irq
  thresh = a
  idx_dirty = np.where(confidence >= thresh)[0]
  for j in range(len(idx_dirty)):
      if idx_dirty[j] in poison_idx:
          tp += 1.0
  fn = len(poison_idx) - tp
  fp = len(idx_dirty) - tp
  tn = len(y_poison) - tp - fp - fn
  return tp,fp,tn,fn