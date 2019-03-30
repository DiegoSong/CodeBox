from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
import pandas as pd


def one_hot(X_train, X_oot, cat_features):
    onehot = OneHotEncoder(handle_unknown='ignore')
    onehot_dict = dict()
    for f in cat_features:
        print(f)
        _one_hot = clone(onehot)
        _one_hot.fit(X_train[[f]])
        onehot_dict[f] = _one_hot
    # train
    for c in cat_features:
        _tmp = onehot_dict[c].transform(X_train[[c]]).toarray()
        _tmp_df = pd.DataFrame(_tmp, columns=[c+ '_' + str(i) for i in onehot_dict[c].categories_[0]])
        del X_train[c]
        X_train = pd.concat([X_train, _tmp_df], axis=1)
    # test
    for c in cat_features:
        _tmp = onehot_dict[c].transform(X_oot[[c]]).toarray()
        _tmp_df = pd.DataFrame(_tmp, columns=[c+ '_' + str(i) for i in onehot_dict[c].categories_[0]])
        del X_oot[c]
        X_oot = pd.concat([X_oot, _tmp_df], axis=1)
    return X_train, X_oot

def count_vector(X_train, X_oot, features):
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), min_df=0.01)
    vector_dict = dict()
    for f in features:
        _vector = clone(vectorizer)
        _vector.fit(X_train[f])
        vector_dict[f] = _vector
    # train    
    for f in features:
        _tmp = vector_dict[f].transform(X_train[[f]]).toarray()
        _tmp_df = pd.DataFrame(_tmp, columns=[f+ '_' + str(i) for i in _vector[f].get_feature_names()])
        del X_train[f]
        X_train = pd.concat([X_train, _tmp_df], axis=1)
    # test    
    for f in features:
        _tmp = vector_dict[f].transform(X_oot[[f]]).toarray()
        _tmp_df = pd.DataFrame(_tmp, columns=[f+ '_' + str(i) for i in _vector[f].get_feature_names()])
        del X_oot[f]
        X_oot = pd.concat([X_oot, _tmp_df], axis=1)
    return X_train, X_oot


oh_features = [u'brand', u'os', u'price', u'size', u'standard', u'pix']
cv_features = [u'active_hash', u'install_hash']
X_train, X_oot = one_hot(X_train, X_oot, oh_features)
X_train, X_oot = count_vector(X_train, X_oot, cv_features)