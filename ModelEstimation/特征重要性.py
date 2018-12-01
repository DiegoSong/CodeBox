import pandas as pd
import numpy as np


def get_importance(opt, X):
    try:
        feature_coef = pd.concat([
            pd.DataFrame(pd.DataFrame(X.columns, columns=['feature_name'])),
            pd.DataFrame(opt.coef_.T, columns=['feature_coef'])
            ],  axis=1)
    except AttributeError:
        feature_coef = pd.concat([
            pd.DataFrame(X.columns, columns=['feature_name']),
            pd.DataFrame(opt.feature_importances_.T, columns=['feature_coef'])
            ],  axis=1)
    feature_coef['abs'] = np.abs(feature_coef['feature_coef'])
    feature_coef = feature_coef.sort_values(by='abs', ascending=False)
    print(feature_coef.sort_values(by='abs', ascending=False)[['feature_name', 'feature_coef']].head(20))
    return feature_coef
