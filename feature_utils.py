import shap
from functools import reduce
import pandas as pd  # package for high-performance, easy-to-use data structures and data analysis
import numpy as np  # fundamental package for scientific computing with Python
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif, RFE
from src.auto_bin_woe import AutoBinWOE
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt  # for plotting
from sklearn.base import clone
import hashlib

from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 100)
sss = StratifiedKFold(n_splits=5, shuffle=True, random_state=45)


def merge_dataframes(dfs, merge_keys, how='inner'):
    dfs_merged = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how=how), dfs)
    return dfs_merged


def genearteMD5(str):
    # 创建md5对象
    hl = hashlib.md5()

    # Tips
    # 此处必须声明encode
    # 否则报错为：hl.update(str)    Unicode-objects must be encoded before hashing
    hl.update(str.encode(encoding='utf-8'))
    return hl.hexdigest()


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
    return feature_coef[['feature_name', 'feature_coef']]


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


def plotcut(pred, y_true, cuts):
    cut = np.percentile(pred, cuts)
    cut = np.append(np.array([float('-Inf')]), cut, axis=0)
    cut = np.append(cut, np.array([float('Inf')]), axis=0)
    result = pd.DataFrame({'y': y_true, 'pred': pd.cut(pred, cut)})
    result['y'].groupby(result['pred']).mean().plot(kind='bar')
    plt.show()


def cal_ks(y_true, y_pred, ks_thre_vali=None, return_thre=False, is_plot=False):
    """

    用于计算ks, 分两种情况：
    1. 已经有从validation上得带的ks threhold
    2. 需要重新计算该数据集本身的threshold
    :param y_true:
    :param y_pred:
    :param ks_thre_vali: 从其他数据集得到了threshold
    :param need_thre: 是否需要返回threshold
    :param is_plot:
    :return:
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    min_score = min(y_pred)
    max_score = max(y_pred)

    max_ks = 0
    bad = len(y_true) - sum(y_true)
    good = sum(y_true)
    ks_thre = 0
    ks_bads = []
    ks_goods = []
    x = []
    if ks_thre_vali is None:
        for i in np.linspace(min_score, max_score, 50):
            val = [[y_pred[j], y_true[j]] for j in range(len(y_pred)) if y_pred[j] < i]
            good_now = sum([val[k][1] for k in range(len(val))])
            bad_now = len(val) - good_now
            if good == 0:
                tmp_ks_good = 0
            else:
                tmp_ks_good = good_now / float(good)
            if bad == 0:
                tmp_ks_bad = 0
            else:
                tmp_ks_bad = bad_now / float(bad)
            ks_now = abs(tmp_ks_good - tmp_ks_bad)
            x.append(i)
            ks_goods.append(tmp_ks_good)
            ks_bads.append(tmp_ks_bad)
            if ks_now > max_ks:
                ks_thre = i
                # ks_good = tmp_ks_good
                # ks_bad = tmp_ks_bad
                max_ks = max(max_ks, ks_now)
    else:
        val = [[y_pred[j], y_true[j]] for j in range(len(y_pred)) if y_pred[j] < ks_thre_vali]
        good_now = sum([val[k][1] for k in range(len(val))])
        bad_now = len(val) - good_now
        if good == 0:
            tmp_ks_good = 0
        else:
            tmp_ks_good = good_now / float(good)
        if bad == 0:
            tmp_ks_bad = 0
        else:
            tmp_ks_bad = bad_now / float(bad)
        ks_now = abs(tmp_ks_good - tmp_ks_bad)
        return ks_now

    if is_plot:
        plt.title('KS curve')
        plt.plot(x, ks_goods, 'g', label='cum good')
        plt.plot(x, ks_bads, 'r', label='cum bad')
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'b--')
        plt.ylim([0, 1])
        # plt.xlim([0, 1])
        plt.ylabel('cumulative population')
        plt.xlabel('scores')
        plt.show()
        # print x
        # print ks_bads
        # print ks_goods
    if return_thre:
        return max_ks, ks_thre
    else:
        return max_ks


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    # auc_scorer = make_scorer(roc_auc_score)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def read_data():
    df = pd.read_csv("../data/qz_dc_feature.csv")
    label = pd.read_csv("../data/qz_dc_label.csv", parse_dates=['create_time'], usecols=['device_md5', 'create_time', 'over_day'])
    label['is_overdue'] = (label['over_day'] > 30) * 1

    df = df.merge(label, how='inner', left_on='imei_md5', right_on='device_md5')

    train = df[(df['create_time'] >= '2017-08-01') & (df['create_time'] < '2017-12-20')].reset_index(drop=True)
    oot = df[df['create_time'] >= '2017-12-20'].reset_index(drop=True)
    X = train.drop(columns=['imei_md5', 'device_md5', 'create_time', 'is_overdue', 'over_day'])
    y = train.is_overdue
    X_oot = oot.drop(columns=['imei_md5', 'device_md5', 'create_time', 'is_overdue', 'over_day'])
    y_oot = oot.is_overdue

    print(train.shape, oot.shape, train['is_overdue'].sum(), oot['is_overdue'].sum())
    return X, y, X_oot, y_oot


def train_by_cv(X, y, X_oot, y_oot, sss, clf, weight=None, **kw):
    pbar = tqdm(total=100)
    auc_train, auc_test, auc_oot = [], [], []
    ks_train, ks_test, ks_oot = [], [], []
    stacking_train = []
    stacking_oot = []
    oos_idx = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if weight is not None:
            clf.fit(X_train, y_train, sample_weight=weight[train_index])
        else:
            clf.fit(X_train, y_train, **kw)
        oos_pred = clf.predict_proba(X_test)[:, 1]
        oot_pred = clf.predict_proba(X_oot)[:, 1]
        oos_idx.extend(test_index)
        stacking_train.extend(oos_pred)
        stacking_oot.append(oot_pred)
        auc_train.append(roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]))
        auc_test.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        auc_oot.append(roc_auc_score(y_oot, clf.predict_proba(X_oot)[:, 1]))
        ks_train.append(cal_ks(y_train, clf.predict_proba(X_train)[:, 1]))
        ks_test.append(cal_ks(y_test, clf.predict_proba(X_test)[:, 1]))
        ks_oot.append(cal_ks(y_oot, clf.predict_proba(X_oot)[:, 1]))
        pbar.update(20)
        
    pbar.close()
    stacking_train = pd.Series(stacking_train, index=oos_idx).sort_index().values
    stacking_oot = np.array(stacking_oot).mean(axis=0)
    print("Train AUC: %s" % np.mean(auc_train))
    print("Test AUC: %s" % np.mean(auc_test))
    print("OOT AUC: %s" % np.mean(auc_oot))
    print("Train KS: %s" % np.mean(ks_train))
    print("Test KS: %s" % np.mean(ks_test))
    print("OOT KS: %s" % np.mean(ks_oot))
    print("--------------------------------------------------- \n")

    return clf, stacking_train, stacking_oot



class XGBEncoder(BaseEstimator, TransformerMixin):
    """
    encode xgb tree leaves as features for linear models
    """

    def __init__(self, params):
        self.xgb_params = params
        self.feature_names = None
        self.xgb = XGBClassifier(**params)
        self.encoder = OneHotEncoder()

    def fit(self, X, y):
        # self.feature_names = list(X.columns)
        self.xgb.fit(X, y)

        # predict data use this model
        xgb_leaves = self.xgb.predict(X, pred_leaf=True)

        # transform the leaves
        self.encoder.fit(xgb_leaves)
        return self

    def transform(self, X, y=None):
        # X = X[self.feature_names]
        xgb_leaves = self.xgb.predict(X, pred_leaf=True)
        return self.encoder.transform(xgb_leaves)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)



class LGBEncoder(BaseEstimator, TransformerMixin):
    """
    encode lgb tree leaves as features for linear models
    """

    def __init__(self, params, min_df=0.001, fname=None):
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '), min_df=min_df)
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


class PCAEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, split=' ', random_state=0, fname=None, min_df=0.0001, **kwd):
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split(split), min_df=min_df)
        self.pca = PCA(n_components=n_components, random_state=random_state, **kwd)
        self.fn = str(fname)
        
    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        vector_features = self.vectorizer.transform(X)
        self.pca.fit(vector_features.toarray())
        return self
        
    def transform(self, X):
        vector_features = self.vectorizer.transform(X)
        new_features = self.pca.transform(vector_features.toarray())
        new_features = pd.DataFrame(new_features, columns=[self.fn + 'PCA%s' % x for x in range(1, 6, 1)])
        return new_features
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class LDAEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, split=' ', random_state=0, fname=None):
        from sklearn.decomposition import LatentDirichletAllocation
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split(split), min_df=0.0002)
        self.lda = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
        self.fn = str(fname)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        vector_features = self.vectorizer.transform(X)
        self.lda.fit(vector_features)
        return self
        
    def transform(self, X):
        vector_features = self.vectorizer.transform(X)
        new_features = self.lda.transform(vector_features)
        new_features = pd.DataFrame(new_features, columns=[self.fn + 'LDA%s' % x for x in range(1, 6, 1)])
        return new_features
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# 特征生成
# 1. PolynomialFeatures
def add_polynomial_features(X, X_test, y=None):
    from sklearn.preprocessing import PolynomialFeatures
    ploy = PolynomialFeatures(2, interaction_only=True)
    ploy.fit(X)
    new_features = ploy.transform(X)
    new_oot_features = ploy.transform(X_test)
    return new_features, new_oot_features


# 4. 重正则化特征
def add_ridge_features(X, X_test, y=None):
    from sklearn.preprocessing import MinMaxScaler
    X = X.fillna(0.0)
    X_test = X_test.fillna(0.0)
    scl = MinMaxScaler().fit(X)
    X = scl.transform(X)
    X_test = scl.transform(X_test)
    
    lr = LogisticRegression(penalty='l1', C=0.02)
    lr.fit(X, y)
    new_features = lr.predict_proba(X)[:, 1]
    new_oot_features = lr.predict_proba(X_test)[:, 1]
    new_features = pd.DataFrame(new_features, columns=['ridge_feature'])
    new_oot_features = pd.DataFrame(new_oot_features, columns=['ridge_feature'])
    return new_features, new_oot_features


# 6. t-SNE特征
def add_tsne_features(X, X_test, y=None):
    from sklearn.manifold import TSNE
    tsne = TSNE()
    X = X.fillna(0.0)
    X_test = X_test.fillna(0.0)
    new_features = tsne.fit_transform(X)
    new_oot_features = tsne.fit_transform(X_test)
    new_features = pd.DataFrame(new_features, columns=['tsne1', 'tsne2'])
    new_oot_features = pd.DataFrame(new_oot_features, columns=['tsne1', 'tsne2'])
    return new_features, new_oot_features


# 7. kmeans特征
def add_kmeans_features(X, X_test, y=None):
    from sklearn.cluster import KMeans
    new_features = pd.DataFrame()
    new_oot_features = pd.DataFrame()
    X = X.fillna(0.0)
    X_test = X_test.fillna(0.0)
    for ncl in range(2, 11):
        cls = KMeans(n_clusters=ncl)
        cls.fit(X)
        _features = cls.predict(X)
        _oot_features = cls.predict(X_test)
        _features = pd.DataFrame(_features, columns=['kmeans_cluster%s' % ncl])
        _oot_features = pd.DataFrame(_oot_features, columns=['kmeans_cluster%s' % ncl])
        new_features = pd.concat([new_features, _features], axis=1)
        new_oot_features = pd.concat([new_oot_features, _oot_features], axis=1)
    return new_features, new_oot_features


def save_feature_model(clf, stacking_train, stacking_oot, X, X_oot, fname):
    oof = pd.concat([stacking_train, stacking_oot])
    oof.to_csv("new_features/%s_oof.csv" % fname, index=0)

    joblib.dump(clf, "new_features/%s_model.pkl" % fname)
    col_str = ",".join(X.columns)
    with open("new_features/%s_model_col.txt" % fname, 'w') as f:
        f.write(col_str)

    X.insert(0, 'imei_md5', stacking_train['imei_md5'])
    X_oot.insert(0, 'imei_md5', stacking_oot['imei_md5'])
    result =  pd.concat([X, X_oot])
    result.to_csv("new_features/%s_features.csv" % fname, index=0)
    

def hyperopt_cv(X_train, y_train, classifier, n_iter=50):
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    # ----- step1: define a objective to minimize -----#
    # 必须是最小化
    def objective(params):
        if classifier == "xgb":
            clf = XGBClassifier(**params)
        elif classifier == "lgb":
            clf = LGBMClassifier(**params)
        elif classifier == "lr":
            clf = LogisticRegression(**params)
        else:
            raise ValueError("classifier type currently not support")
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        train_scores = []
        cv_scores = []
        for train_index, test_index in skf.split(X_train, y_train):
            X_tr, X_vali = X_train.iloc[train_index, :], X_train.iloc[test_index, :]
            y_tr, y_vali = y_train[train_index], y_train[test_index]
            clf.fit(X_tr, y_tr)
            y_train_pred = clf.predict_proba(X_tr)[:, 1]
            y_vali_pred = clf.predict_proba(X_vali)[:, 1]
            train_scores.append(roc_auc_score(y_tr, y_train_pred))
            cv_scores.append(roc_auc_score(y_vali, y_vali_pred))

        # cv performance
        cv_score = np.mean(cv_scores)
        # cv stability
        cv_std = np.std(cv_scores)
        # train vs. cv differences
        diff = sum([abs(train_scores[i] - cv_scores[i]) for i in range(len(train_scores))])
        # objective: high cv score + low cv std + low train&cv diff
        loss = (1 - cv_score) + cv_std + diff
        pbar.update()
        return {
            'loss': loss,
            'status': STATUS_OK,
            # -- store other results like this
            'eval_time': time.time(),
            'other_stuff': {'type': None, 'value': [0, 1, 2]},
            #         # -- attachments are handled differently
            #         'attachments':
            #             {'time_module': pickle.dumps(time.time)}
        }

    # ----- step2: define a search space -----#
    # search_space = {}
    if classifier == "xgb":
        search_space = {
            'n_estimators': hp.choice('n_estimators', np.arange(50, 200, dtype=int)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
            # A problem with max_depth casted to float instead of int with
            # the hp.quniform method.
            'max_depth': hp.choice('max_depth', np.arange(3, 6, dtype=int)),
            "max_delta_step": hp.quniform('max_delta_step', 0, 20, 1),
            'min_child_weight': hp.quniform('min_child_weight', 0, 100, 1),
            'subsample': hp.uniform('subsample', 0.2, 1.0),
            'gamma': hp.uniform('gamma', 0.1, 50),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
            "reg_lambda": hp.uniform('reg_lambda', 0.1, 100),
            "reg_alpha": hp.uniform('reg_alpha', 0.1, 100),
            "scale_pos_weight": hp.loguniform('scale_pos_weight', 1, 50),
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            # Increase this number if you have more cores. Otherwise, remove it and it will default
            # to the maxium number.
            'nthread': 6,
            'booster': 'dart',
            'tree_method': hp.choice("tree_method", ['exact', "approx"]),
            'silent': True,
            'seed': 42
        }
    elif classifier == "lr":
        search_space = {
            'class_weight': hp.choice("class_weight", ["balanced", None]),
            "C": hp.uniform("C", 1e-3, 1e2),
            'penalty': hp.choice("penalty", ["l1", "l2"]),
        }
    elif classifier == "lgb":
        search_space = {
            'learning_rate': hp.uniform("learning_rate", 0.01, 0.2),
            'num_leaves': hp.choice('num_leaves', np.arange(8, 50, dtype=int)),
            # 'max_depth': (0, 5),
            'min_child_samples': hp.choice('min_child_samples', np.arange(20, 200, dtype=int)),
            # 'max_bin': (100, 1000),
            'subsample': hp.uniform('subsample', 0.1, 1.0),
            'subsample_freq': hp.choice("subsample_freq", np.arange(0, 10, dtype=int)),
            'colsample_bytree': hp.uniform("colsample_bytree", 0.01, 1.0),
            'min_child_weight': hp.uniform("min_child_weight", 1e-3, 10),
            # 'subsample_for_bin': (100000, 500000),
            'reg_lambda': hp.uniform("reg_lambda", 1.0, 1000),
            'reg_alpha': hp.uniform("reg_alpha", 1.0, 100),
            'scale_pos_weight': hp.uniform("scale_pos_weight", 1.0, 50),
            'n_estimators': hp.choice('n_estimators', np.arange(50, 200, dtype=int)),
        }
    elif classifier == "rf":
        search_space = {
            'n_estimators': hp.choice("n_estimators", np.arange(100, 1000, dtype=int)),
            "max_depth": hp.choice('max_depth', np.arange(3, 5, dtype=int)),
            'min_samples_split': hp.choice("min_samples_split", np.arange(2, 200, dtype=int)),
            "max_features": hp.uniform("max_features", 0.2, 1.0),
        }
    else:
        raise ValueError("classifier type currently not support")

    # ----- step3: define a trial object for tracking -----#
    trials = Trials()
    pbar = tqdm(total=n_iter, desc="Hyperopt")
    # ----- step4: optimization by fmin -----#
    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=trials)
    pbar.close()
    best_params = space_eval(search_space, best)
    best_clf = None
    if classifier == "lr":
        best_clf = LogisticRegression(**best_params).fit(X_train, y_train)
    elif classifier == "xgb":
        best_clf = XGBClassifier(**best_params).fit(X_train, y_train)
    elif classifier == "lgb":
        best_clf = LGBMClassifier(**best_params).fit(X_train, y_train)
    elif classifier == "rf":
        best_clf = RandomForestClassifier(**best_params).fit(X_train, y_train)

    return best_params, best_clf

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
        _tmp = vector_dict[f].transform(X_train[f]).toarray()
        _tmp_df = pd.DataFrame(_tmp, columns=[f+ '_' + str(i) for i in vector_dict[f].get_feature_names()])
        del X_train[f]
        X_train = pd.concat([X_train, _tmp_df], axis=1)
    # test    
    for f in features:
        _tmp = vector_dict[f].transform(X_oot[f]).toarray()
        _tmp_df = pd.DataFrame(_tmp, columns=[f+ '_' + str(i) for i in vector_dict[f].get_feature_names()])
        del X_oot[f]
        X_oot = pd.concat([X_oot, _tmp_df], axis=1)
    return X_train, X_oot

def min_max(X_train, X_oot):
    fn = X_train.columns
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_oot = scaler.transform(X_oot)
    X_train = pd.DataFrame(X_train, columns=fn)
    X_oot = pd.DataFrame(X_oot, columns=fn)
    return X_train, X_oot, scaler


# 特征选择
def feature_select(X_train, y_train, method='iv', kb=100, rfe=30):
    if method == 'iv':
        method = mutual_info_classif
    elif method == 'f':
        method = f_classif
    
    # chi2
    fn = X_train.columns
    selector1 = SelectKBest(chi2, kb)
    selector1.fit(X_train, y_train)
    
    # information value
    selector2 = SelectKBest(method, kb)
    selector2.fit(X_train, y_train)
    left_features = list(set(fn[selector2.get_support()].tolist() + fn[selector1.get_support()].tolist()))

    # RFE
    _X_tmp = X_train[left_features]
    fn = _X_tmp.columns
    clf = LogisticRegression(penalty='l2', C=0.2)
    selector = RFE(estimator=clf, n_features_to_select=rfe)
    selector.fit(_X_tmp, y_train)
    
    left_features = fn[selector.get_support()].tolist()
    X_train = X_train[left_features]
    return left_features


# 检验oot和train与y的关系是否一致单调
def oot_monotonicity_check(X_train, y_train, X_oot, y_oot):
    from src.auto_bin_woe import AutoBinWOE
    woe = AutoBinWOE(bins=5)
    woe.fit(X_train, y_train)
    X_woe = woe.transform(X_train)
    X_oot_woe = woe.transform(X_oot)
    dict_dock = woe.compute_bad_rate(X_train, y_train)
    dict_dock_oot = woe.compute_bad_rate(X_oot, y_oot)
    
    cols = []
    for k in dict_dock:
        corr = np.corrcoef(dict_dock[k]['bad_rate'], dict_dock_oot[k]['bad_rate'])[0][1]
        dict_dock[k]['bad_rate_oot'] = dict_dock_oot[k]['bad_rate']
        mean1 = dict_dock[k]['bad_rate'].mean()
        mean2 = dict_dock[k]['bad_rate_oot'].mean()
        if corr > 0:
        # if (mean2 - dict_dock[k]['br'].values[0]) * (mean1 - dict_dock[k]['bad_rate'].values[0])>0:
            cols.append(k)
            print(k)
            print(dict_dock[k])
     
    br_df = pd.DataFrame()
    for f in dict_dock:
        tmp = dict_dock[f]
        tmp.insert(0, 'feature', f)
        br_df = pd.concat([br_df, tmp])
    return cols, br_df