from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


class StackingAveragedModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, n_folds=5):
        self.base_model1 = LogisticRegression()
        self.base_model2 = CatBoostClassifier()
        self.stacking_model = LGBMClassifier()
        self.base_models_1 = []
        self.base_models_2 = []
        self.categorical_feats = ['']
        self.X1col = {}
        self.X2col = {}
        self.n_folds = n_folds

    # 我们将原来的模型clone出来，并且进行实现fit功能
    def fit(self, X1, X2, X3, X4, X5, y):
        '''
        X1: dc_feature
        X2: CatBoost feature
        X3: likelihood feature
        X4: PCA active
        X5: PCA pkg_flags
        '''
        X1 = X1.astype(X1col)
        X1 = X1[self.X1col.keys]
        
        X2 = X2.astype(X2col)
        X2 = X2[self.X2col.keys]

        X3 = X3.astype(X3col)
        X3 = X3[self.X3col.keys]

        self.meta_model_ = clone(self.meta_model)
        # kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=100)
        sfolder = StratifiedKFold(n_splits = self.n_folds, random_state=100, shuffle=True)
        cat_feats_idx = np.unique([np.argmax(X2.columns == feat) for feat in self.categorical_feats])
        
        #对于每个模型，使用交叉验证的方法来训练初级学习器，并且得到次级训练集
        out_of_fold_predictions = np.zeros((X1.shape[0], 2))

        for train_index, holdout_index in sfolder.split(X1, y):
            instance = clone(self.base_model1)
            self.base_models_1.append(instance)
            instance.fit(X1.iloc[train_index, :], y[train_index])
            y_pred = instance.predict_proba(X1.iloc[holdout_index, :])[:,1]
            out_of_fold_predictions[holdout_index, 0] = y_pred
        
        for train_index, holdout_index in sfolder.split(X2, y):
            instance = clone(self.base_model2)
            self.base_models_2.append(instance)
            instance.fit(X2.iloc[train_index, :], y[train_index], cat_features = cat_feats_idx)
            y_pred = instance.predict_proba(X2.iloc[holdout_index, :])[:,1]
            out_of_fold_predictions[holdout_index, 1] = y_pred
        
        
        # 使用次级训练集来训练次级学习器
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

#在上面的fit方法当中，我们已经将我们训练出来的初级学习器和次级学习器保存下来了
#predict的时候只需要用这些学习器构造我们的次级预测数据集并且进行预测就可以了
    def predict_proba(self, X1, X2, X3, X4, X5,):
        X1 = X1.astype(float)
        X1 = X1[self.X1col]
        
        X2 = X2.astype()
        X2 = X2[self.X2col]
        tmp1 = np.zeros((X1.shape[0], 5))
        tmp2 = np.zeros((X1.shape[0], 5))
        i, j = 0, 0
        for model in self.base_models_1:
            tmp1[:,i] = model.predict_proba(X1)[:,1]
            i += 1

        for model in self.base_models_2:
            tmp2[:,j] = model.predict_proba(X2)[:,1]
            j += 1
        meta_features = pd.DataFrame()
        meta_features['lr'] = tmp1.mean(axis=1)
        meta_features['cat'] = tmp2.mean(axis=1)
        return self.meta_model_.predict_proba(meta_features)[:,1]

#ss = StackingAveragedModels()
#ss.fit(X1,X2,y)
#ss.predict_proba(X1,X2)
