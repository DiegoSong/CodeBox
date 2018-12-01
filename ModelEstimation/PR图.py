import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import auc 
import matplotlib.pyplot as plt


def check_threshold(y_true, y_hat):
    tp = sum((y_true==1) & (y_hat==1)) * 1.0
    tn = sum((y_true==0) & (y_hat==0)) * 1.0
    fp = sum((y_true==0) & (y_hat==1)) * 1.0
    fn = sum((y_true==1) & (y_hat==0)) * 1.0
    precision = tp/(fp+tp+1)
    recall = tp/(fn+tp+1)
    return precision, recall


def rate_0_1(y_true):
    return float(sum(y_true==1)) / (len(y_true) + 1)
    

thresholds = np.linspace(0.0,1.0,30)

scores = []
for threshold in np.linspace(0.0, 1.0, 30):
    data['y_hat'] = (data['probability'] > threshold) * 1.0
    p, r = check_threshold(data['label'], data['y_hat'])
    scores.append([p, r])
    # bin_data = data[data['probability']>threshold]
    # precision, recall, thresholds = check_threshold(bin_data['label'], bin_data['y_hat'])
    # scores.append([threshold, sum(bin_data['label']==1), rate_0_1(bin_data['label'])])
scores = np.array(scores)
plt.figure(figsize=(18, 8))
plt.plot(thresholds, scores[:, 1], label='recall')
plt.plot(thresholds, scores[:, 0], label='precision')
plt.xticks(np.linspace(0.0,1.0,6))
plt.legend()
plt.show()