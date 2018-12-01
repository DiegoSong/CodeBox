import shap
import pandas as pd
from lightgbm import LGBMClassifier
import numpy as np


def shap_feature_select(clf, X, k=None):
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    shape_importance = pd.concat([pd.DataFrame(X.columns, columns=['feature_name']),
                                  pd.DataFrame(np.sum(np.abs(shap_values), axis=0), columns=['importance'])], axis=1)
    shape_importance['idx'] = shape_importance.index
    shape_importance = shape_importance.sort_values(['importance'], ascending=False).reset_index(drop=True)
    if k:
        value_cols = shape_importance.loc[:k, :]['feature_name']
    else:
        value_cols = shape_importance[shape_importance['importance']>0]['feature_name']
    return value_cols


# train XGBoost model
X, y = shap.datasets.boston()
clf = LGBMClassifier()
clf.fit(X, y)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")
