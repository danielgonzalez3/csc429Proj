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
from model import detect_clean_data
import random

BASEDIR_MNIST = "./MNIST"

############################################################################################
# Compute the ROC (Receiver Operator Characteristics) for the RBF Outlier Detection Method #
############################################################################################
# predictions_train = np.load(os.path.join(BASEDIR_MNIST,'predictions_anomaly_train.npy'))
alphas = np.linspace(0,5,30)
results = []
IRQS = []
results_IRQS= []
i = 0
anomaly_poison = MNISTModel(anomalyDetector=True)
for p in probs:
  print(p)
  x_train_poison,y_train_poison,poisoned_idx_train = PoisonMNIST(X=x_train,
                                                Y = y_train,
                                                p=p)
  anomaly_poison.load(os.path.join(BASEDIR_MNIST,'poison'+str(p)+'.h5'))
  x_train_backdoor = x_train_poison[poisoned_idx_train]
  y_train_backdoor = y_train_poison[poisoned_idx_train]
  indices = np.arange(y_train_poison.shape[0])
  cleanIdx = np.delete(indices,poisoned_idx_train,axis=0)
  x_train_clean = x_train_poison[cleanIdx]
  y_train_clean = y_train_poison[cleanIdx]
  predictions = anomaly_poison.model.predict(x_train_poison)
  confidence = predictions[np.arange(predictions.shape[0]),np.argmax(y_train_poison,axis=1)]
  irq = stats.iqr(confidence)
  q3 = np.quantile(confidence, 0.75)
  thresh = q3 + 1.5*irq
  IRQS.append(thresh)
  for a in alphas:
    tp_rbf,fp_rbf,tn_rbf,fn_rbf = detect_clean_data(predictions,poisoned_idx_train,x_train_poison,y_train_poison,a)
    results.append([tp_rbf,fp_rbf,tn_rbf,fn_rbf])
  tp_rbf,fp_rbf,tn_rbf,fn_rbf = detect_clean_data(predictions,poisoned_idx_train,x_train_poison,y_train_poison,thresh)
  results_IRQS.append([tp_rbf,fp_rbf,tn_rbf,fn_rbf])
  i += 1
np.save(os.path.join(BASEDIR_MNIST,'results.npy'),results)

##########################
# MNIST Poisoning Attack #
##########################
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
probs = [0.03,0.04,0.05,0.06,0.07, 0.08,0.1,0.13,0.15,0.25]
correctness_all_test_data = []
correctness_poison_test_data = []
true_positives = []
false_positives = []
false_negatives = []
true_negatives = []
correctness_all_train_data = []
predictions_train = np.zeros((len(probs),60000,10))
restart = False
i = 0
various_ps = []
x_test_poison,y_test_poison,poisoned_idx = PoisonMNIST(X=x_test,
                                                Y = y_test,
                                                p=0.1)
x_backdoor = x_test_poison[poisoned_idx]
y_backdoor = y_test_poison[poisoned_idx]
for p in probs:
  # generate poison data
  x_train_poison,y_train_poison,poisoned_idx_train = PoisonMNIST(X=x_train,
                                                Y = y_train,
                                                p=p)
  x_train_backdoor = x_train_poison[poisoned_idx_train]
  y_train_backdoor = y_train_poison[poisoned_idx_train]
  indices = np.arange(y_train_poison.shape[0])
  cleanIdx = np.delete(indices,poisoned_idx_train,axis=0)
  x_train_clean = x_train_poison[cleanIdx]
  y_train_clean = y_train_poison[cleanIdx]
  print('Number of poisoned:',len(poisoned_idx_train))

  softmax_poison = MNISTModel(RBF=False)
  if not restart and os.path.isfile(os.path.join(BASEDIR_MNIST,'model'+str(p)+'.h5')):
    softmax_poison.load(os.path.join(BASEDIR_MNIST,'model'+str(p)+'.h5'))
  else:
    softmax_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(BASEDIR_MNIST,'model'+str(p)+'.h5'),epochs=5)
  predictions = np.argmax(softmax_poison.predict(x_backdoor),axis=1)
  labels = np.argmax(y_backdoor,axis=1)
  acc_reg_poison = np.sum(predictions == labels)/len(labels)
  print('Poison success',acc_reg_poison)
  predictions = np.argmax(softmax_poison.predict(x_train_poison),axis=1)
  labels = np.argmax(y_train_poison,axis=1)
  acc_train_reg = np.sum(predictions == labels)/len(labels)

  predictions = np.argmax(softmax_poison.predict(x_test_poison),axis=1)
  labels = np.argmax(y_test_poison,axis=1)
  acc_reg_clean = np.sum(predictions == labels)/len(labels)
  print('Test accuracy',acc_reg_clean)

  # perform activation clustering defense
  classifier = KerasClassifier(model=softmax_poison.model)
  defence = ActivationDefence(classifier, x_train_poison, y_train_poison)
  report, is_clean_lst = defence.detect_poison(nb_clusters=2,
                                              nb_dims=10,
                                              reduce="PCA")
  # calculate tp,fp,tn,fn
  attempt_idx = np.sort(np.where(np.array(is_clean_lst)== 0)[0])
  poison_idx_sort = np.sort(poisoned_idx_train)
  # calculate true positive
  tp_reg = 0.0
  for idx in attempt_idx:
      if idx in poison_idx_sort:
          tp_reg += 1.0
  fn_reg = len(poison_idx_sort) - tp_reg
  fp_reg = len(attempt_idx) - tp_reg
  tn_reg = y_train_poison.shape[0] - fn_reg - fp_reg - tp_reg

  print('Results (tp,fp,tn,fn)',tp_reg,fp_reg,tn_reg,fn_reg)

  # train anomaly on feature space of poisoned model
  anomaly_poison = MNISTModel(anomalyDetector=True)
  if not restart and os.path.isfile(os.path.join(BASEDIR_MNIST,'poison'+str(p)+'.h5')):
    anomaly_poison.load(os.path.join(BASEDIR_MNIST,'poison'+str(p)+'.h5'))
  else:
    anomaly_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(BASEDIR_MNIST,'poison'+str(p)+'.h5'),epochs=10)
  predictions = np.argmax(anomaly_poison.predict(x_backdoor),axis=1)
  labels = np.argmax(y_backdoor,axis=1)
  acc_rbf_poison = np.sum(predictions == labels)/len(labels)
  print('Poison success',acc_rbf_poison)

  predictions = np.argmax(anomaly_poison.predict(x_test_poison),axis=1)
  labels = np.argmax(y_test_poison,axis=1)
  acc_rbf_clean = np.sum(predictions == labels)/len(labels)
  print('Test accuracy',acc_rbf_clean)

  confidence = anomaly_poison.predict(x_train_poison)
  predictions_train[i,:,:] = confidence
  i += 1
  predictions = np.argmax(confidence,axis=1)
  labels = np.argmax(y_train_poison,axis=1)
  acc_train_rbf = np.sum(predictions == labels)/len(labels)

  for quantile in [0.5,0.6,0.7,0.8,0.9]:
    tp_rbf,fp_rbf,tn_rbf,fn_rbf = detect_clean_data(anomaly_poison, poisoned_idx_train, x_train_poison,y_train_poison,quantile)
    various_ps.append([tp_rbf,fp_rbf,tn_rbf,fn_rbf])
  print('Results (tp,fp,tn,fn)',tp_rbf,fp_rbf,tn_rbf,fn_rbf)

  correctness_all_test_data.append([acc_reg_clean,acc_rbf_clean])
  correctness_poison_test_data.append([acc_reg_poison,acc_rbf_poison])
  true_positives.append([tp_reg,tp_rbf])
  false_positives.append([fp_reg,fp_rbf])
  false_negatives.append([fn_reg,fn_rbf])
  true_negatives.append([tn_reg,tn_rbf])
  correctness_all_train_data.append([acc_train_reg,acc_train_rbf])

np.save(os.path.join(BASEDIR_MNIST,'tp.npy'),true_positives)
np.save(os.path.join(BASEDIR_MNIST,'fn.npy'),false_negatives)
np.save(os.path.join(BASEDIR_MNIST,'tn.npy'),true_negatives)
np.save(os.path.join(BASEDIR_MNIST,'fp.npy'),false_positives)
np.save(os.path.join(BASEDIR_MNIST,'acc_clean.npy'),correctness_all_test_data)
np.save(os.path.join(BASEDIR_MNIST,'acc_poison.npy'),correctness_poison_test_data)
np.save(os.path.join(BASEDIR_MNIST,'acc_train.npy'),correctness_all_train_data)
np.save(os.path.join(BASEDIR_MNIST,'predictions_anomaly_train.npy'),predictions_train)
np.save(os.path.join(BASEDIR_MNIST,'various_ps.npy'),various_ps)

######################
## Compute ROC Curve #
######################
predictions_train = np.load(os.path.join(BASEDIR_MNIST,'predictions_anomaly_train.npy'))
alphas = np.linspace(0,5,30)
results = []
IRQS = []
results_IRQS= []
i = 0
anomaly_poison = MNISTModel(anomalyDetector=True)
for p in probs:
  print(p)
  x_train_poison,y_train_poison,poisoned_idx_train = PoisonMNIST(X=x_train,
                                                Y = y_train,
                                                p=p)
  anomaly_poison.load(os.path.join(BASEDIR_MNIST,'poison'+str(p)+'.h5'))
  x_train_backdoor = x_train_poison[poisoned_idx_train]
  y_train_backdoor = y_train_poison[poisoned_idx_train]
  indices = np.arange(y_train_poison.shape[0])
  cleanIdx = np.delete(indices,poisoned_idx_train,axis=0)
  x_train_clean = x_train_poison[cleanIdx]
  y_train_clean = y_train_poison[cleanIdx]
  predictions = anomaly_poison.model.predict(x_train_poison)
  confidence = predictions[np.arange(predictions.shape[0]),np.argmax(y_train_poison,axis=1)]
  irq = stats.iqr(confidence)
  q3 = np.quantile(confidence, 0.75)
  thresh = q3 + 1.5*irq
  IRQS.append(thresh)
  for a in alphas:
    tp_rbf,fp_rbf,tn_rbf,fn_rbf = detect_clean_data(predictions,poisoned_idx_train,x_train_poison,y_train_poison,a)
    results.append([tp_rbf,fp_rbf,tn_rbf,fn_rbf])
  tp_rbf,fp_rbf,tn_rbf,fn_rbf = detect_clean_data(predictions,poisoned_idx_train,x_train_poison,y_train_poison,thresh)
  results_IRQS.append([tp_rbf,fp_rbf,tn_rbf,fn_rbf])
  i += 1
np.save(os.path.join(BASEDIR_MNIST,'results.npy'),results)




