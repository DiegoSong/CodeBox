from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd


class LikelihoodEncoder(BaseEstimator, TransformerMixin):
    """
    仅可用于分类变量
    """
    def __init__(self, c=30.0):
        np.random.seed(1232345)
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=21387)
        self.ll_map = {}
        self.c = c

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            pass
        else:
            raise TypeError("X must be pandas DataFrame with category features.")

        for i in range(X.shape[1]):
            feature = X.iloc[:, i]
            fname = feature.name
            global_avg = np.mean(y)
            raw_data = pd.concat([X, pd.DataFrame(y, columns=['is_overdue'])], axis=1).reset_index(drop=True)
            raw_data = raw_data.fillna(-999)
            feats_likeli = raw_data.groupby(fname)['is_overdue'].agg({np.sum, 'count'})
            feats_likeli.columns = ['sum', 'count']
            feats_likeli = feats_likeli.reset_index()

            feats_likeli[fname + '_likeli'] = (feats_likeli['sum'] + self.c * global_avg) / (feats_likeli['count'] + self.c)
            feats_likeli[fname + '_likeli'] = feats_likeli[fname + '_likeli'].fillna(global_avg)
            fdict = {}
            for k, v in feats_likeli[[fname, fname + '_likeli']].values:
                fdict[k] = v
            self.ll_map[fname] = fdict

    def transform(self, X):
        """

        :param X:
        :return:
        """
        transformed_df = pd.DataFrame()
        for i in range(X.shape[1]):
            feature = X.iloc[:, i]
            fname = feature.name
            fdict = self.ll_map[fname]
            feature = feature.fillna(-999)
            transformed_feature = []
            for f in feature:
                try:
                    transformed_feature.append(fdict[f])
                except KeyError as e:
                    transformed_feature.append(0.0)
            # transformed_feature = feature.apply(lambda x: fdict[x])
            transformed_df = pd.concat([transformed_df,
                                        pd.DataFrame(transformed_feature, columns=[fname + '_likeli'])], axis=1)
        return transformed_df

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
