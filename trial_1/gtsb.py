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
from model import RBFLayer, softargmax, RBF_Loss, RBF_Soft_Loss, DistanceMetric, ResNetLayer, ResNetV1, PoisonGTSB, detect_clean_data, load_split
import random

GTSB_ZIP_PATH = "./GTSB-german-traffic-sign.zip"
BASEDIR_GTSB = "./gtsb-german-traffic-sign"
RBF_LAMBDA = 0.5

# Open the zip file
if len(os.listdir(BASEDIR_GTSB)) == 0:
    with zipfile.ZipFile(GTSB_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(BASEDIR_GTSB)
        print("ZIP file extracted successfully.")
else:
    print("The extraction path is not empty. Skipping the extraction.")

# derive the path to the training and testing CSV files
trainPath = os.path.join(BASEDIR_GTSB, "Train.csv")
testPath = os.path.join(BASEDIR_GTSB, "Test.csv")
if not os.path.isfile(os.path.join(BASEDIR_GTSB,'x_train.npy')):
  # load the training and testing data
  print("[INFO] loading training and testing data...")
  (trainX, trainY) = load_split(BASEDIR_GTSB, trainPath)
  (testX, testY) = load_split(BASEDIR_GTSB, testPath)
  trainX = trainX.astype("float32") / 255.0
  testX = testX.astype("float32") / 255.0
  np.save(os.path.join(BASEDIR_GTSB,'x_train.npy'),trainX)
  np.save(os.path.join(BASEDIR_GTSB,'y_train.npy'),trainY)
  np.save(os.path.join(BASEDIR_GTSB,'x_test.npy'),testX)
  np.save(os.path.join(BASEDIR_GTSB,'y_test.npy'),testY) 
else:
  trainX = np.load(os.path.join(BASEDIR_GTSB,'x_train.npy'))*255
  trainY = np.load(os.path.join(BASEDIR_GTSB,'y_train.npy'))
  testX = np.load(os.path.join(BASEDIR_GTSB,'x_test.npy'))*255
  testY = np.load(os.path.join(BASEDIR_GTSB,'y_test.npy')) 
# one-hot encode the training and testing labels
numLabels = len(np.unique(trainY))
trainY = keras.utils.to_categorical(trainY, numLabels)
testY = keras.utils.to_categorical(testY, numLabels)
# account for skew in the labeled data
classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals

print('Number of training instances',trainY.shape[0])
print('Number of testing instances',testY.shape[0])
print('Number of classes',numLabels)

#############################
# Poison the GTSB Data Set  #
#############################
x_train_poison,y_train_poison,x_train_clean,y_train_clean,x_train_backdoor,y_train_backdoor,idx_poison = PoisonGTSB(trainX,trainY,targetLabel=5,p=0.005)
x_test_poison,y_test_poison,x_test_clean,y_test_clean,x_test_backdoor,y_test_backdoor,idx_poison  = PoisonGTSB(testX,testY,targetLabel=5)

img = x_train_backdoor[8]

cv2.imwrite('./content/clean_german_sign.png',x_train_backdoor[8]*255)
plt.savefig('./content/poison_german_sign.png')

probs = [0.005,0.01,0.02,0.03,0.04]
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
x_test_poison,y_test_poison,x_test_clean,y_test_clean,x_test_backdoor,y_test_backdoor,idx_poison  = PoisonGTSB(testX,testY,targetLabel=5,p=0.1)
for p in probs:
  # generate poison data
  x_train_poison,y_train_poison,x_train_clean,y_train_clean,x_train_backdoor,y_train_backdoor,poisoned_idx_train = PoisonGTSB(trainX,trainY,targetLabel=5,p=p)
  print('Number of poisoned:',len(poisoned_idx_train))
  ################################################################################################################### softmax train
  softmax_poison = ResNetV1(RBF=False)
  if not restart and os.path.isfile(os.path.join(BASEDIR_GTSB,'model'+str(p)+'.h5')):
    softmax_poison.load(os.path.join(BASEDIR_GTSB,'model'+str(p)+'.h5'))
  else:
    softmax_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(BASEDIR_GTSB,'model'+str(p)+'.h5'),epochs=4)
  ##################################################################################################################### get predictions
  predictions = np.argmax(softmax_poison.predict(x_test_backdoor),axis=1)
  labels = np.argmax(y_test_backdoor,axis=1)
  acc_reg_poison = np.sum(predictions == labels)/len(labels)
  print('Poison success - acc_reg_poison',acc_reg_poison)
  predictions = np.argmax(softmax_poison.predict(x_train_poison),axis=1)
  labels = np.argmax(y_train_poison,axis=1)
  acc_train_reg = np.sum(predictions == labels)/len(labels)

  predictions = np.argmax(softmax_poison.predict(x_test_poison),axis=1)
  labels = np.argmax(y_test_poison,axis=1)
  acc_reg_clean = np.sum(predictions == labels)/len(labels)
  print('Test accuracy - acc_reg_clean', acc_reg_clean)

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


  ################################################################################################################################### train anomaly
  anomaly_poison = ResNetV1(anomalyDetector=True)
  if not restart and os.path.isfile(os.path.join(BASEDIR_GTSB,'poison'+str(p)+'.h5')):
    anomaly_poison.load(os.path.join(BASEDIR_GTSB,'poison'+str(p)+'.h5'))
  else:
    anomaly_poison.train(x_train_poison,y_train_poison,saveTo=os.path.join(BASEDIR_GTSB,'poison'+str(p)+'.h5'),epochs=4)
  ##################################################################################################################### get predictions

  predictions = np.argmax(anomaly_poison.predict(x_test_backdoor),axis=1)
  labels = np.argmax(y_test_backdoor,axis=1)
  acc_rbf_poison = np.sum(predictions == labels)/len(labels)
  print('Poison success - acc_rbf_poison',acc_rbf_poison)

  predictions = np.argmax(anomaly_poison.predict(x_test_poison),axis=1)
  labels = np.argmax(y_test_poison,axis=1)
  acc_rbf_clean = np.sum(predictions == labels)/len(labels)
  print('Test accuracy - acc_rbf_clean',acc_rbf_clean)

  confidence = anomaly_poison.predict(x_train_poison)
  # predictions_train[i,:,:] = confidence
  i += 1
  predictions = np.argmax(confidence,axis=1)
  labels = np.argmax(y_train_poison,axis=1)
  acc_train_rbf = np.sum(predictions == labels)/len(labels)
  tp_rbf,fp_rbf,tn_rbf,fn_rbf = detect_clean_data(confidence, poisoned_idx_train, x_train_poison,y_train_poison,1.5)

  correctness_all_test_data.append([acc_reg_clean,acc_rbf_clean])
  correctness_poison_test_data.append([acc_reg_poison,acc_rbf_poison])
  true_positives.append([tp_reg,tp_rbf])
  false_positives.append([fp_reg,fp_rbf])
  false_negatives.append([fn_reg,fn_rbf])
  true_negatives.append([tn_reg,tn_rbf])
  correctness_all_train_data.append([acc_train_reg,acc_train_rbf])

np.save(os.path.join(BASEDIR_GTSB,'tp.npy'),true_positives)
np.save(os.path.join(BASEDIR_GTSB,'fn.npy'),false_negatives)
np.save(os.path.join(BASEDIR_GTSB,'tn.npy'),true_negatives)
np.save(os.path.join(BASEDIR_GTSB,'fp.npy'),false_positives)
np.save(os.path.join(BASEDIR_GTSB,'acc_clean.npy'),correctness_all_test_data)
np.save(os.path.join(BASEDIR_GTSB,'acc_poison.npy'),correctness_poison_test_data)
np.save(os.path.join(BASEDIR_GTSB,'acc_train.npy'),correctness_all_train_data)
np.save(os.path.join(BASEDIR_GTSB,'predictions_anomaly_train.npy'),predictions_train)
np.save(os.path.join(BASEDIR_GTSB,'various_ps.npy'),various_ps)


