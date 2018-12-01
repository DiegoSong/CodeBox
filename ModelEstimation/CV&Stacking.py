from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import numpy as np

clf = LGBMClassifier(max_depth=-1, learning_rate=0.1, n_estimators=50, class_weight='balanced',
                     reg_lambda=50, num_leaves=8, n_jobs=4, boosting_type='dart')


def train_by_cv(X, y, X_oot, y_oot, sss, clf, weight=None, **kw):
    r = 1
    auc_train, auc_test, auc_oot = [], [], []
    ks_train, ks_test, ks_oot = [], [], []
    stacking_train = []
    stacking_oot = []
    oos_idx = []
    for train_index, test_index in sss.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        elif isinstance(X, np.ndarray):
            X_train, X_test = X[train_index, :], X[test_index, :]
        else:
            X = pd.DataFrame(X)
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
        print("%s/5" % r)
        r += 1

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


# 保存stacking oof
stacking_train = pd.concat([train['id_num_biz'][test_idx].reset_index(drop=True),
                        pd.DataFrame(stacking_train, columns=['op_history'])], axis=1)

stacking_oot = pd.concat([oot['id_num_biz'].reset_index(drop=True),
                        pd.DataFrame(stacking_oot, columns=['op_history'])], axis=1)
oof = pd.concat([stacking_train, stacking_oot])
oof.to_csv("oof_data/op_history_oof.csv", index=False)