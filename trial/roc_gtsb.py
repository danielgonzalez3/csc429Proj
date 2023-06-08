import sklearn
import numpy as np
import os
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

BASEDIR_GTSB = "./gtsb-german-traffic-sign"

results = np.load(os.path.join(BASEDIR_GTSB,'results.npy'))
results = np.array(results)
probs = [0.005,0.01,0.02,0.03,0.04]
alphas = np.linspace(0,5,30)
TP = results[:,0]
FP = results[:,1]
TN = results[:,2]
FN = results[:,3]
TP_rate = TP / (TP + FN)
FP_rate = FP / (TN + FP)
plt.figure(figsize=(8,7))
plt.subplot(2,1,2)
for i in range(len(probs)):
  tp_rate = TP_rate[i*len(alphas):(i+1)*len(alphas)]
  fp_rate = FP_rate[i*len(alphas):(i+1)*len(alphas)]
  n = int(35856*probs[i])
  auc = sklearn.metrics.auc(fp_rate,tp_rate)/np.ptp(fp_rate)
  label = 'n='+str(n)+' AUC='+str(np.round(auc,2))
  plt.scatter(fp_rate,tp_rate,label=label,alpha=0.8)
plt.xlabel('FP Rate',fontsize=14)
plt.ylabel('TP Rate',fontsize=14)
plt.title('RBF Outlier Detection Method on GTSB Attack',fontsize=16,pad=10)
plt.legend(prop={'size': 9})
plt.grid(True)
plt.tight_layout(pad=3)
plt.tick_params(axis='both',labelsize = 12)
plt.subplot(2,1,1)
plt.savefig('./images/ROC2.eps', format='eps', dpi=1000)
