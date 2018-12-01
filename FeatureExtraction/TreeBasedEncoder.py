from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from lightgbm import LGBMClassifier


class LGBEncoder(BaseEstimator, TransformerMixin):
    """
    encode lgb tree leaves as features for linear models
    """

    def __init__(self, params, min_df=0.001, fname=None):
        self.vectorizer = CountVectorizer(tokenizer=spliter, min_df=min_df)
        self.lgb_params = params
        self.feature_names = None
        self.lgb = LGBMClassifier(**params)
        self.encoder = OneHotEncoder()
        self.fn = fname
        self.n_result = params['n_estimators'] * params['num_leaves']

    def fit(self, X, y):
        self.vectorizer.fit(X)
        vector_features = self.vectorizer.transform(X)
        self.lgb.fit(vector_features.toarray(), y)
        print(vector_features.shape)
        # predict data use this model
        lgb_leaves = self.lgb.predict(vector_features.toarray(), pred_leaf=True)

        # transform the leaves
        self.encoder.fit(lgb_leaves)
        return self

    def transform(self, X, y=None):
        vector_features = self.vectorizer.transform(X)
        lgb_leaves = self.lgb.predict(vector_features.toarray(), pred_leaf=True)
        result = self.encoder.transform(lgb_leaves).toarray()
        return pd.DataFrame(result, columns=[self.fn + 'LGB%s' % x for x in range(1, self.n_result+1, 1)])

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)