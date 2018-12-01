# -*- coding: utf-8 -*-
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDClassifier
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
from mlxtend.plotting import plot_confusion_matrix
from mlxtend.evaluate import confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from skopt.space import Real, Categorical, Integer
import lightgbm as lgb
from catboost import CatBoostClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

today = time.strftime("%Y-%m-%d")


def get_raw_data(raw_path, report_path, label="y"):
    try:
        raw = pd.read_csv(raw_path, error_bad_lines=False)
    except Exception as e:
        print(e)
        return

    with open(report_path, "a+") as fo:
        fo.write("dataset info:\n\n")
        fo.write("modeling date:" + str(today) + "\n")
        fo.write("dataset shape:" + str(raw.shape) + "\n")
        if label in raw.columns:
            fo.write("overall default rate:" + str(raw[label].mean()) + "\n")
        fo.write("=" * 100 + "\n")

    return raw


# TODO: reset_index很重要！！！！！
def dataset_split(raw, report_path, timestamp, method="oot", random_seed=42):
    if method == "oot":
        raw = raw.sort_values(by="create_time")
        if timestamp is not None:
            train_df = raw[raw['create_time'] <= timestamp].reset_index(drop=True)
            test_df = raw[raw["create_time"] > timestamp].reset_index(drop=True)
        else:
            train_index = int(0.8 * len(raw))
            train_df = raw.iloc[:train_index, :].reset_index(drop=True)
            test_df = raw.iloc[train_index:, :].reset_index(drop=True)
        del train_df["create_time"], test_df["create_time"]
    else:
        if "create_time" in raw.columns:
            del raw["create_time"]
        train_df, test_df = train_test_split(raw, stratify=raw["y"], random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    with open(report_path, "a+") as fo:
        fo.write("dataset split\n\n")
        fo.write("数据集切分方式：{}\n".format(method))
        if method == "oos":
            fo.write("random seed:{}\n".format(random_seed))
        fo.write("timestamp:{}\n".format(timestamp))
        fo.write("train set shape:{}, train y=1 ratio:{}\n".format(train_df.shape, train_df["y"].mean()))
        # fo.write("validation set shape:{}, validation y=1 ratio:{}\n".format(validation_df.shape,
        #                                                                      validation_df["y"].mean()))
        fo.write("test set shape:{}, test y=1 ratio:{}\n".format(test_df.shape, test_df["y"].mean()))
        fo.write("=" * 100 + "\n")
    return train_df, test_df


def data_prepare(train_df, test_df, label="y"):
    y_train = train_df[label]
    del train_df[label]
    X_train = train_df

    y_test = test_df[label]
    del test_df[label]
    X_test = test_df

    return X_train, y_train, X_test, y_test


def cal_correlation(df, label, type="pearson"):
    corr_val_list = []
    if type == "pearson":
        for c in df.columns:
            if c == label:
                continue
            p = df[c].corr(df[label])
            corr_val_list.append({"feature_name": c, "pearson": p, "abs_pearson": abs(p)})
        corr_df = pd.DataFrame(corr_val_list).sort_values(by="abs_pearson", ascending=False).reset_index(drop=True)
    return corr_df


def get_weekday(x, time_pattern):
    weekday = datetime.datetime.strptime(str(x), time_pattern).weekday()
    weekday += 1
    return weekday


def is_weekend(x, time_pattern):
    weekday = datetime.datetime.strptime(str(x), time_pattern).weekday()
    weekday += 1
    if weekday == 6 or weekday == 7:
        return 1
    else:
        return 0


def time_segmentation(x, start_pos, end_pos=None):
    try:
        if end_pos is None:
            x = str(x)[start_pos:]
        else:
            x = str(x)[start_pos:end_pos]
        x = int(x)
    except Exception as e:
        print(e, x)
    # define 9-19 as work time
    if x >= 8 and x < 19:
        return "worktime"
    elif x >= 19 and x <= 24:
        return "resttime"
    else:
        return "raretime"


def get_confusion_matrix(y_true, y_pred, p=50, t=None):
    if t is None:
        threshold = np.percentile(y_pred, p)
    else:
        threshold = t
    y_labels = [1 if i > threshold else 0 for i in y_pred]
    cm = confusion_matrix(y_target=y_true,
                          y_predicted=y_labels)
    plot_confusion_matrix(conf_mat=cm)


def smote_sample(X_train, y_train, minority_class, ratio, sample_kind='regular'):
    # X_train=np.array(X_train)
    y_train = np.array(y_train).astype(int)
    # print(X_train, y_train)
    minority_cnt = sum([1 if i == minority_class else 0 for i in y_train])
    # 用minority样本比例反推样本个数
    target_sample = int(float(len(X_train) * ratio - minority_cnt) / (1 - ratio))
    # print('sample size', minority_cnt, target_sample)
    target_sample += int(minority_cnt)
    ratio_dict = {minority_class: target_sample}
    print(ratio_dict)
    sm = SMOTE(ratio=ratio_dict, kind=sample_kind, random_state=42)
    # sm=ADASYN(ratio=ratio_dict, random_state=42)
    X_res, y_res = sm.fit_sample(X_train, y_train)
    return X_res, y_res


def under_sample(ori_df, ratio, target_label=0):
    # 原始set的长度
    total_len = len(ori_df)
    # 目标label df长度
    ori_target_df = ori_df[ori_df["y"].astype(int) == target_label]
    ori_target_len = len(ori_target_df)
    # 需要删除的条数
    remove_size = int((ori_target_len - ratio * total_len) / (1 - ratio))
    # 采样条数
    sample_size = ori_target_len - remove_size
    sample_target_df = ori_target_df.sample(n=sample_size)
    non_target_df = ori_df[ori_df["y"].astype(int) == (1 - target_label)]
    # 将采样后的df与未采样的合起来
    new_df = sample_target_df.append(non_target_df)
    if "showtime" in new_df.columns:
        new_df = new_df.sort_values(by="showtime")
        del new_df["showtime"]
    else:
        new_df = shuffle(new_df)
    return new_df


def model_selection_cv(X_train, y_train, classifier="xgb", pos_weight=1.0):
    if classifier == "xgb":
        params = {"learning_rate": [0.1],
                  "n_estimators": [200],
                  "colsample_bytree": [0.5, 0.8],
                  "reg_alpha": [10, 50],
                  "max_depth": [3, 4],
                  "booster": ["dart"],
                  "scale_pos_weight": [1, pos_weight],
                  "subsample": [0.5, 0.8],
                  "gamma": [10, 30]
                  }
        classifier = XGBClassifier(objective="binary:logistic", n_jobs=6)
        grad = GridSearchCV(classifier, params, cv=5, n_jobs=6, scoring="roc_auc", verbose=1).fit(X_train,
                                                                                                  y_train,
                                                                                                  eval_metric="auc"
                                                                                                  )
    elif classifier == "lgb":
        params = {"num_leaves": [16, 32],
                  "min_child_samples": [50, 100],
                  "learning_rate": [0.1, 0.2],
                  "n_estimators": [200],
                  "colsample_bytree": [0.4, 0.8],
                  "subsample": [0.8],
                  "reg_alpha": [10, 50, 100],
                  "class_weight": ["balanced", None],
                  "boosting_type": ["dart"],
                  }
        classifier = LGBMClassifier(objective="binary", n_jobs=6)
        grad = GridSearchCV(classifier, params, cv=5, n_jobs=6, scoring="roc_auc", verbose=1).fit(X_train,
                                                                                                  y_train,
                                                                                                  eval_metric="auc"
                                                                                                  )
    elif classifier == "lr":
        params = {"C": [0.01, 0.05, 0.008, 0.1, 0.5],
                  "penalty": ["l1"],
                  "class_weight": [None, "balanced"]}
        classifier = LogisticRegression()
        grad = GridSearchCV(classifier, params, cv=5, n_jobs=6, scoring="roc_auc", verbose=1).fit(X_train,
                                                                                                  y_train)
    elif classifier == "rf":
        params = {"n_estimators": [200, 300, 500], "max_depth": [3, 4, 5], "max_features": [0.4, 0.6, 0.8],
                  "min_sample_split": [5, 20, 50]}
        grad = GridSearchCV(RandomForestClassifier(n_jobs=6), params, cv=5, n_jobs=6, scoring="roc_auc",
                            verbose=1).fit(X_train, y_train)
    elif classifier == "sgd":
        params = {"penalty": ["l1", "elasticnet"], "alpha": [1e-2, 0.08, 0.1], "tol": [1e-3, 1e-4],
                  "class_weight": ["balanced", None]}
        grad = GridSearchCV(SGDClassifier(loss="log", n_jobs=6), params, cv=5, n_jobs=6,
                            scoring="roc_auc",
                            verbose=1).fit(X_train, y_train)
    else:
        raise TypeError("model type currently not support")
    clf = grad.best_estimator_
    best_score = grad.best_score_
    best_params = grad.best_params_
    # with open(report_path, "a+") as fo:
    #     fo.write("model selection results:\n\n")
    #     fo.write("cv auc:" + str(best_score) + "\n")
    #     fo.write("best params:" + str(best_params) + "\n")
    #     fo.write("=" * 100 + "\n")
    return clf, best_params, best_score


def hyperopt_cv(X_train, y_train, classifier, iter=50):
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

    # ----- step4: optimization by fmin -----#
    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=iter,
        trials=trials)
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


def bayes_cv(X_train, y_train, classifier, cat_feats_idx=None):
    ITERATIONS = 80
    if classifier == "xgb":
        bayes_cv_tuner = BayesSearchCV(
            estimator=xgb.XGBClassifier(
                n_jobs=6,
                objective='binary:logistic',
                eval_metric='auc',
                silent=True,
                tree_method='approx'
            ),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'min_child_weight': (0, 100),
                'max_depth': (3, 5),
                'max_delta_step': (0, 20),
                'subsample': (0.2, 1.0, 'uniform'),
                'colsample_bytree': (0.2, 1.0, 'uniform'),
                # 'colsample_bylevel': (0.2, 1.0, 'uniform'),
                'reg_lambda': (1.0, 100, 'log-uniform'),
                'reg_alpha': (1.0, 100, 'log-uniform'),
                'gamma': (1.0, 50, 'log-uniform'),
                'n_estimators': (50, 200),
                'scale_pos_weight': (1.0, 50, 'log-uniform'),
                "tree_method": ()
            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=6,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )
    if classifier == "lgb":
        bayes_cv_tuner = BayesSearchCV(
            estimator=lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                boosting_type="dart",
                n_jobs=6,
                verbose=0
            ),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'num_leaves': (8, 50),
                # 'max_depth': (0, 5),
                'min_child_samples': (20, 200),
                # 'max_bin': (100, 1000),
                'subsample': (0.1, 1.0, 'uniform'),
                'subsample_freq': (0, 10),
                'colsample_bytree': (0.01, 1.0, 'uniform'),
                'min_child_weight': (1e-3, 10),
                # 'subsample_for_bin': (100000, 500000),
                'reg_lambda': (1.0, 1000, 'log-uniform'),
                'reg_alpha': (1.0, 100, 'log-uniform'),
                'scale_pos_weight': (1.0, 50, 'log-uniform'),
                'n_estimators': (50, 100),
            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=6,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )
    if classifier == "sgd":
        bayes_cv_tuner = BayesSearchCV(
            estimator=SGDClassifier(
                loss='log',
                n_jobs=6,
                penalty="elasticnet",
                learning_rate="optimal",
                class_weight="balanced"
            ),
            search_spaces={
                'alpha': (0, 1.0, 'log-uniform'),
                'tol': (1e-6, 1e-3, 'log-uniform'),
            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=6,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )
    if classifier == "rf":
        bayes_cv_tuner = BayesSearchCV(
            estimator=RandomForestClassifier(
                n_jobs=6,
                class_weight="balanced"
            ),
            search_spaces={
                'n_estimators': (100, 1000),
                "max_depth": (3, 5),
                'min_samples_split': (2, 200),
                "max_features": (0.2, 1.0, "uniform"),

            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=6,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )
    if classifier == "lr":
        bayes_cv_tuner = BayesSearchCV(
            estimator=LogisticRegression(
                n_jobs=6,
            ),
            search_spaces={
                'class_weight': Categorical(["balanced", None]),
                "C": (1e-3, 1e2, 'log-uniform'),
                'penalty': Categorical(["l1", "l2"]),
            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=6,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )

    if classifier == "catb":
        bayes_cv_tuner = BayesSearchCV(
            estimator=CatBoostClassifier(
                loss_function="Logloss"
            ),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                "depth": (3, 5),
                'iterations': (50, 300),
                "l2_leaf_reg": (1e-3, 1e3, 'log-uniform'),
                "subsample": (0.2, 1.0, 'log-uniform'),
                "scale_pos_weight": (1.0, 100, 'uniform'),
            },
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            ),
            n_jobs=6,
            n_iter=ITERATIONS,
            verbose=0,
            refit=True,
            random_state=42
        )
    if classifier == "catb":
        cv = bayes_cv_tuner.fit(X_train, y_train, cat_features=cat_feats_idx)
    else:
        cv = bayes_cv_tuner.fit(X_train, y_train,
                                # callback=status_print
                                )
    best_score = cv.best_score_
    best_params = cv.best_params_
    clf = cv.best_estimator_

    return best_score, best_params, clf


def gbdt_model_evaluation(X_train, y_train, X_validation, y_validation, n_estimators, learning_rate, subsample,
                          min_samples_split, max_depth, max_features):
    tmp_clf = GradientBoostingClassifier(n_estimators=int(n_estimators), learning_rate=learning_rate,
                                         subsample=subsample, max_features=max_features,
                                         min_samples_split=int(min_samples_split), max_depth=int(max_depth)).fit(
        X_train,
        y_train)
    y_predict = tmp_clf.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, y_predict)
    return auc


def lr_model_evaluation(X_train, y_train, X_validation, y_validation, C, max_iter):
    # weight_dict = {1: class_weight, 0: 1 - class_weight}
    tmp_clf = LogisticRegression(penalty="l1", class_weight="balanced", C=C, max_iter=int(max_iter)).fit(X_train,
                                                                                                         y_train)
    y_predict = tmp_clf.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, y_predict)
    return auc


def xgb_model_evaluation(X_train, y_train, X_validation, y_validation, num_boost_round, eta, max_depth, subsample,
                         colsample_bytree, alpha):
    params = {"eta": eta, "max_depth": int(max_depth), "subsample": subsample, "colsample_bytree": colsample_bytree,
              "alpha": alpha, "objective": "binary:logistic", "booster": "dart", "nthread": 6}

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns)
    dvali = xgb.DMatrix(X_validation, label=y_validation, feature_names=X_train.columns)
    tmp_clf = xgb.train(params, dtrain, num_boost_round=int(num_boost_round))
    y_predict = tmp_clf.predict(dvali)
    auc = roc_auc_score(y_validation, y_predict)
    return auc


def sklearn_xgb_evaluation(X_train, y_train, X_validation, y_validation, learning_rate, colsample_bytree, reg_alpha,
                           max_depth, n_estimators, scale_pos_weight, subsample, gamma):
    clf = XGBClassifier(learning_rate=learning_rate, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                        max_depth=int(max_depth), objective="binary:logistic", scale_pos_weight=scale_pos_weight,
                        subsample=subsample, gamma=gamma, booster="dart", n_estimators=int(n_estimators), n_jobs=6).fit(
        X_train,
        y_train)
    y_pred = clf.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, y_pred)
    return auc


def el_model_evaluation(X_train, y_train, X_validation, y_validation, alpha, l1_ratio, tol):
    tmp_clf = SGDClassifier(loss="log", penalty="elasticnet", alpha=alpha, l1_ratio=l1_ratio, tol=tol, n_jobs=6).fit(
        X_train, y_train)
    y_predict = tmp_clf.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, y_predict)
    return auc


def rf_model_evaluation(X_train, y_train, X_validation, y_validation, n_estimators, min_samples_split):
    tmp_clf = RandomForestClassifier(n_estimators=int(n_estimators), min_samples_split=int(min_samples_split),
                                     n_jobs=6).fit(X_train, y_train)
    y_predict = tmp_clf.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, y_predict)
    return auc


def model_selection_bayes_opt(X, y, model_type="gbdt", scale_pos_weight=1.0):
    train_index = int(0.8 * len(X))
    X_train, y_train = X.iloc[:train_index, :], y[:train_index]
    X_validation, y_validation = X.iloc[train_index:, :], y[train_index:]

    init_points = 5
    num_iter = 50
    if model_type == "gbdt":
        gbdt_partial = partial(gbdt_model_evaluation, X_train, y_train, X_validation, y_validation)
        BO = BayesianOptimization(gbdt_partial, {"n_estimators": (100, 150),
                                                 "learning_rate": (0.1, 0.2),
                                                 "subsample": (0.5, 1.0),
                                                 "min_samples_split": (5, 10),
                                                 "max_depth": (3, 5),
                                                 "max_features": (0.5, 1.0)})

    if model_type == "lr":
        lr_partial = partial(lr_model_evaluation, X_train, y_train, X_validation, y_validation)
        BO = BayesianOptimization(lr_partial, {  # "tol": (1e-6, 1e-4),
            "C": (0.0001, 0.1),
            # "class_weight": (0.5, 0.8),
            "max_iter": (100, 300)})

    if model_type == "el":
        el_partial = partial(el_model_evaluation, X_train, y_train, X_validation, y_validation)
        BO = BayesianOptimization(el_partial, {"alpha": (1e-4, 0.1),
                                               "l1_ratio": (0.15, 1.0),
                                               "tol": (1e-4, 1e-3)})
    # if model_type == "xgb":
    #     xgb_partial = partial(xgb_model_evaluation, X_train, y_train, X_validation, y_validation)
    #     BO = BayesianOptimization(xgb_partial, {
    #         # "num_boost_round": (100, 200),
    #         "eta": (0.1, 0.2),
    #         "max_depth": (3, 5),
    #         "subsample": (0.5, 1.0),
    #         "colsample_bytree": (0.5, 1.0),
    #         "alpha": (1.0, 10),
    #         "gamma": (0.1, 10)})

    if model_type == "xgb":
        xgb_partial = partial(sklearn_xgb_evaluation, X_train, y_train, X_validation, y_validation)
        BO = BayesianOptimization(xgb_partial, {
            "learning_rate": (0.1, 0.2),
            "colsample_bytree": (0.5, 1.0),
            "reg_alpha": (1.0, 20),
            "max_depth": (3, 5),
            "n_estimators": (100, 200),
            "scale_pos_weight": (1.0, scale_pos_weight),
            "subsample": (0.5, 1.0),
            "gamma": (0.1, 20)})

    if model_type == "rf":
        rf_partial = partial(rf_model_evaluation, X_train, y_train, X_validation, y_validation)
        BO = BayesianOptimization(rf_partial, {"n_estimators": (200, 600),
                                               "min_samples_split": (5, 20)})

    BO.maximize(init_points=init_points, n_iter=num_iter)
    best_val = BO.res["max"]["max_val"]
    best_params = BO.res["max"]["max_params"]
    if model_type == "xgb":
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])
        # clf = XGBClassifier(learning_rate=best_params["learning_rate"],
        #                     colsample_bytree=best_params["colsample_bytree"], reg_alpha=best_params["reg_alpha"],
        #                     max_depth=best_params["max_depth"], objective="binary:logistic",
        #                     scale_pos_weight=best_params["scale_pos_weight"],
        #                     subsample=best_params["subsample"], gamma=best_params["gamma"], booster="dart",
        #                     n_estimators=best_params["n_estimators"],
        #                     n_jobs=6).fit(X, y)
    return best_val, best_params


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="roc_auc")
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


def validation_learning_curve(estimator, title, X, y, ylim=None, train_sizes=np.linspace(.1, 1.0, 5), params=None):
    train_index = int(0.8 * len(X))
    try:
        X_train, X_validation = X.iloc[:train_index, :], X.iloc[train_index:, :]
    except:
        X_train, X_validation = X[:train_index, :], X[train_index:, :]
    y_train, y_validation = y[:train_index], y[train_index:]
    train_score = []
    validation_score = []
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    # if estimator == "xgb":
    #     X_validation = xgb.DMatrix(X_validation, label=y_validation, feature_names=X_train.columns)
    if params is not None:
        params["max_depth"] = int(params["max_depth"])
    for train_ratio in train_sizes:
        train_index = int(train_ratio * len(X_train))
        try:
            X_train_tmp = X_train.iloc[:train_index, :]
        except:
            X_train_tmp = X_train[:train_index, :]
        y_train_tmp = y_train[:train_index]
        # if estimator == "xgb":
        #     X_train_tmp = xgb.DMatrix(X_train_tmp, label=y_train_tmp, feature_names=X_train.columns)
        #     tmp_clf = xgb.train(params, X_train_tmp, num_boost_round=int(params["num_boost_round"]))
        # else:
        #     tmp_clf = estimator.fit(X_train_tmp, y_train_tmp)
        try:
            y_train_pred = estimator.predict_proba(X_train_tmp)[:, 1]
            y_validation_pred = estimator.predict_proba(X_validation)[:, 1]
        except Exception as e:
            y_train_pred = estimator.predict(X_train_tmp)
            y_validation_pred = estimator.predict(X_validation)
        train_score.append(roc_auc_score(y_train_tmp, y_train_pred))
        validation_score.append(roc_auc_score(y_validation, y_validation_pred))

    plt.plot(train_sizes, train_score, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, validation_score, 'o-', color="g",
             label="validation score")

    plt.legend(loc="best")
    return plt


def find_breakeven_point(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # print(len(precision), len(recall), len(thresholds))
    min_diff = None
    index = 0
    for i in range(len(precision) - 1):
        p = precision[i]
        r = recall[i]
        t = thresholds[i]
        diff = abs(p - r)
        if min_diff is None:
            min_diff = diff
        elif (diff < min_diff) and (p != 0) and (r != 0):
            min_diff = diff
            index = i
    return precision[index], recall[index], thresholds[index]


def plot_precision_recall(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')


def plotcut(pred, y_true, cuts=[20, 40, 60, 80, 90, 95]):
    cut = np.percentile(pred, cuts)
    cut = np.append(np.array([float('-Inf')]), cut, axis=0)
    cut = np.append(cut, np.array([float('Inf')]), axis=0)
    result = pd.DataFrame({'y': y_true, 'pred': pd.cut(pred, cut)})
    result['y'].groupby(result['pred']).mean().plot(kind='bar')
    plt.show()
    result['y'].groupby(result['pred']).count().plot(kind='bar')
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
        plt.ylim([0, 1])
        # plt.xlim([0, 1])
        plt.ylabel('cumulative population')
        plt.xlabel('scores')
        # plt.show()
        plt.savefig("ks_plot.png")
    if return_thre:
        return max_ks, ks_thre
    else:
        return max_ks


def get_importance(clf, feature_names):
    try:
        if isinstance(clf, XGBClassifier) or isinstance(clf, LGBMClassifier) or isinstance(clf, RandomForestClassifier):
            importance = pd.DataFrame(
                {"feature_names": feature_names, "importance": list(clf.feature_importances_)})
            importance = importance.sort_values(by="importance", ascending=False).reset_index(drop=True)

        elif isinstance(clf, LogisticRegression) or isinstance(clf, SGDClassifier):
            importance = pd.DataFrame(
                {"feature_names": feature_names, "coef": list(clf.coef_[0])})
            importance["abs_coef"] = importance["coef"].abs()
            importance = importance.sort_values(by="abs_coef", ascending=False).reset_index(drop=True)
        return importance
    except Exception as e:
        print(e)


if __name__ == '__main__':
    df = pd.read_csv("../../results/wanka_pass_xgb_predictions_2018-08-15.csv")
    print(cal_ks(y_true=1.0 - df["y_true"], y_pred=1.0 - df["y_pred"], return_thre=True, is_plot=True))
