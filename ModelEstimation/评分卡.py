import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# precision, recall, thresholds = precision_recall_curve(data['label'], data['probability'])
# auc(precision, recall)


def plotcut(pred, y_true, cuts):
    cut=np.percentile(pred, cuts)
    cut=np.append(np.array([float('-Inf')]), cut, axis=0)
    cut=np.append(cut, np.array([float('Inf')]), axis=0)
    result = pd.DataFrame({'y': y_true, 'pred': pd.cut(pred, cut)})
    result['y'].groupby(result['pred']).mean().plot(kind='bar')
    plt.show()
    # result['y'].groupby(result['pred']).count().plot(kind='bar')
    # plt.show()

plotcut(data['probability'], data['label'], cuts=[10,20,30,40,50,60,70,80,90])