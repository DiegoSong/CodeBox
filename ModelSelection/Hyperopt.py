from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import time


def hyperopt_cv(X_train, y_train, classifier, n_iter=50):
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