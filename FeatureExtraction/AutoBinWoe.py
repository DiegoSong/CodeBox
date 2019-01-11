from scipy import stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils.multiclass import type_of_target
from copy import deepcopy


class WOE:
    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        # X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = np.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1
        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')


    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max


class AutoBinWOE(object):

    def __init__(self, bins=5):
        self.threshold = None
        self.bins = bins
        self.spearman_coeffient = dict()
        self.dict_dock = dict()
        self.data_matrix = dict()
        self.continuous_col = []
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def _discrete_single_fit(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: threshold dict
        '''
        threshold = {}
        for i in range(self.bins):
            point1 = stats.scoreatpercentile(x, i * (100/self.bins))
            point2 = stats.scoreatpercentile(x, (i + 1) * (100/self.bins))
            threshold[i] = [point1, point2]
        return threshold

    def _discrete_fit(self, X):
        thresholds = {}
        for col in X.columns:
            x = X[col].values
            x_type = type_of_target(x)
            if x_type in ['continuous', 'multiclass']:
                self.continuous_col.append(col)
                thr = self._discrete_single_fit(x)
                thresholds[col] = thr
        self.threshold = thresholds

    def _discrete_transform(self, X):
        # TODO: 阈值区间双闭会有部分重叠
        tmp = []
        for col in X.columns:
            x = X[col].values
            if col in self.continuous_col:
                thr = self.threshold[col]
                res = X[col].copy()
                for k in thr.keys():
                    point1, point2 = thr[k]
                    x1 = x[np.where((x >= point1) & (x <= point2))]
                    mask = np.in1d(x, x1)
                    res[mask] = k
                tmp.append(res)
            else:
                tmp.append(x)
        return pd.DataFrame(np.array(tmp).T, columns=X.columns)

    def _discrete_fit_transform(self, X):
        self._discrete_fit(X)
        return self._discrete_transform(X)

    def compute_bad_rate(self, X, y):
        dict_dock = deepcopy(self.dict_dock)
        for col in X.columns:
            x = X[col].values
            if col in self.continuous_col:
                thr = self.dict_dock[col]
                bad_rate = []
                for k in thr.index:
                    point1, point2 = thr.loc[k, 'left'], thr.loc[k, 'right']
                    x1 = x[np.where((x >= point1) & (x <= point2))]
                    mask = np.in1d(x, x1)
                    bad_rate.append(y[mask].sum() / y[mask].count())
                dict_dock[col]['bad_rate'] = bad_rate
            else:
                bad_rate = []
                thr = self.dict_dock[col]
                for k in thr.index:
                    x1 = x[np.where(x == k)]
                    mask = np.in1d(x, x1)
                    bad_rate.append(y[mask].sum() / y[mask].count())
                dict_dock[col]['bad_rate'] = bad_rate
        return dict_dock

    def _category_feature(self, x, y):
        data = pd.DataFrame({"var": x, "label": list(y)})
        data_matrix = data.groupby("var")["label"].agg(
            {'bad': np.count_nonzero, 'obs': np.size})
        data_matrix["good"] = data_matrix["obs"] - data_matrix["bad"]
        data_matrix["good_rate"] = data_matrix["good"] / data_matrix["obs"]
        data_matrix["bad_rate"] = data_matrix["bad"] / data_matrix["obs"]
        data_matrix["lower"] = data_matrix.index.tolist()
        data_matrix["left"] = None
        data_matrix["right"] = None
        coef = self.spearman(data_matrix)
        return coef, data_matrix

    def monotony_single_fit(self, x, y, threshold):
        """
        对cut labels 进行合并迭代,寻找使得abs(spearman coeffient)最大的labels组合
        :param X: 原始数据列向量, pd.Series or np.array
        :param y: list or np.array or pd.Series
        :return: new cut label and the best spearman coeffient
        """

        data = pd.DataFrame({"var": x, "label": list(y)})
        data_matrix_1 = data.groupby("var")["label"].agg(
            {'bad': np.count_nonzero, 'obs': np.size})

        data_matrix = data_matrix_1.copy()
        threshold = pd.DataFrame([[i[0], i[1]] for i in threshold.values()], index=threshold.keys(),
                                 columns=['left', 'right'])
        data_matrix = data_matrix.merge(threshold, how='inner', left_index=True, right_index=True)

        data_matrix["lower"] = data_matrix.index.tolist()
        coef = self.spearman(data_matrix)
        new_label = list(np.copy(data_matrix.index))
        data_matrix_best = data_matrix.copy()
        while np.float64(coef) < 0.9999:
            coef_list = []
            label_list = []
            data_matrix_list = []
            for i in range(len(new_label) - 2):
                label_temp = list(np.copy(new_label))
                label_temp.remove(new_label[i + 1])
                data_matrix_temp = data_matrix_best.copy()
                series_index = data_matrix_temp.index
                data_matrix_temp["bad"].loc[series_index[i + 1]] += data_matrix_temp["bad"].loc[series_index[i]]
                data_matrix_temp["obs"].loc[series_index[i + 1]] += data_matrix_temp["obs"].loc[series_index[i]]
                data_matrix_temp["lower"].loc[series_index[i + 1]] = data_matrix_temp["lower"].loc[series_index[i]]
                data_matrix_temp["left"].loc[series_index[i + 1]] = data_matrix_temp["left"].loc[series_index[i]]
                data_matrix_temp = data_matrix_temp.drop(data_matrix_temp.index[i])
                data_matrix_list.append(data_matrix_temp)
                coef_list.append(self.spearman(data_matrix_temp))
                label_list.append(label_temp)
            index = np.argmax(coef_list)
            coef = coef_list[index]
            new_label = label_list[index]
            data_matrix_best = data_matrix_list[index]
        data_matrix_best.index = np.array(["bin_" + str(i) for i in range(data_matrix_best.shape[0])])

        data_matrix_best["good"] = data_matrix_best["obs"] - data_matrix_best["bad"]
        data_matrix_best["good_rate"] = data_matrix_best["good"] / data_matrix_best["obs"]
        data_matrix_best["bad_rate"] = data_matrix_best["bad"] / data_matrix_best["obs"]
        return coef, data_matrix_best

    def fit(self, X, y):
        X_bin = self._discrete_fit_transform(X)
        for variable in tqdm(X_bin.columns):
            if variable in self.threshold.keys():
                thr = self.threshold[variable]
                coef, data_matrix = self.monotony_single_fit(X_bin[variable], y, thr)
            else:
                coef, data_matrix = self._category_feature(X_bin[variable], y)

            self.spearman_coeffient.update({variable: coef})
            data_matrix_woe = self.calc_woe(data_matrix)
            data_matrix_woe['left'] = data_matrix['left']
            data_matrix_woe['right'] = data_matrix['right']
            self.dict_dock.update({variable: data_matrix_woe})
            self.data_matrix.update({variable: data_matrix})

    def transform(self, X):
        for variable in X.columns:
            self.threshold[variable] = {}
            for k in self.dict_dock[variable].index:
                thr = [self.dict_dock[variable].loc[k, 'left'], self.dict_dock[variable].loc[k, 'right']]
                self.threshold[variable][k] = thr

        X_bin = self._discrete_transform(X)
        return self._woe_replace(X_bin)

    @staticmethod
    def spearman(x):
        """
        计算spearman coeffient
        :param x: pd.DataFrame contains number of bad, observition, lower, upper of cut labels
        :return: abs(spearman coeffient)
        """
        bad = x["bad"]
        count = x["obs"]
        lower = x["lower"]
        count = [1 if x == 0 else x for x in count]
        bad_rate = bad / count
        try:
            cor, pval = stats.spearmanr(a=bad_rate, b=lower, axis=0)
        except ValueError:
            print("bad_rate is %f" % bad_rate)
            print("bad is %s" % repr(bad))
            print("lower is %s" % repr(lower))
        return abs(cor)

    def calc_woe(self, data):
        """
        :param data: DataFrame(Var:float,bad:int,good:int)
        :return: weight of evidence
        """

        if 'bad' not in data.columns:
            raise ValueError("data columns don't has 'bad' column")
        if 'good' not in data.columns:
            raise ValueError("data columns don't has 'good' column")

        data['woe'] = np.log((data['bad'] / data['good']) / (data['bad'].sum() / data['good'].sum()))
        data['woe'] = data['woe'].replace(-np.inf, self._WOE_MIN)
        data['woe'] = data['woe'].replace(np.inf, self._WOE_MAX)
        return data[['woe']]

    def _woe_replace(self, X):
        """
        :param X: pd.DataFrame after discretion
        :return: pd.DataFrame with result of woe transform
        """
        keys = self.dict_dock.keys()
        if len(keys) < 0:
            return pd.DataFrame(np.zeros(X.shape), columns=X.columns)
        else:
            arg = [(X, name) for name in X.columns if name in keys]
            result_temp = [self._get_woe_transform(elem) for elem in arg]
            result = [elem for elem in result_temp if elem is not None]
            if len(result) == 0:
                return pd.DataFrame(np.zeros(X.shape), columns=X.columns)
            else:
                try:
                    return pd.DataFrame(np.transpose(result), columns=X.columns)
                except ValueError:
                    print("woe_transform error")
                    print(result)

    def _get_woe_transform(self, arg):
        X, name = arg
        result_temp = pd.Series([self._woe_elem_transform(col_name=name, elem=elem) for elem in X[name]])
        return result_temp

    def _woe_elem_transform(self, col_name, elem):
        matrix = self.dict_dock.get(col_name)
        if elem in list(matrix.index):
            return matrix.loc[elem, "woe"]
        else:
            return 0.0


def nan_replace(x, method=None):
    """

    :param x:
    :param method:
    :return:
    """
    if method is not None and method not in ["median", "mean"]:
        raise ValueError("method must be one of %s " % repr(["median", "mean"]))
    replace_element = 0.0
    if method == "median":
        replace_element = np.nanmedian(x)
    elif method == "mean":
        replace_element = np.nanmean(x)
    return x.fillna(replace_element)


if __name__ == '__main__':
    from src.feature_utils import *
    df = pd.read_csv("data/op32.csv")
    df['op_lasts'] = df['op_lasts'] * -1
    df = df.fillna(-9999)
    df.loc[(df['account_balance'] < -9999) | (df['account_balance'] > 10000), 'account_balance'] = -9999

    oot = '2017-11-25'
    label = pd.read_excel("data/label_user_idnum.xlsx", parse_dates=['create_time'], sheet='sheet1')
    df = df.merge(label[['id_num_biz', 'is_overdue', 'create_time']], how='inner', on=['id_num_biz'])

    oot_train = df[df['create_time'] < oot].reset_index(drop=True)
    oot_test = df[df['create_time'] >= oot].reset_index(drop=True)

    X = oot_train.drop(['id_num_biz', 'is_overdue', 'create_time'], axis=1)
    y = oot_train['is_overdue']
    X_oot = oot_test.drop(['id_num_biz', 'is_overdue', 'create_time'], axis=1)
    y_oot = oot_test['is_overdue']

    woe = AutoBinWOE(bins=10)
    woe.fit(X, y)
    X_woe = woe.transform(X)
    X_oot_woe = woe.transform(X_oot)
    dict_dock = woe.compute_bad_rate(X, y)

    clf = LogisticRegression(C=0.2, penalty='l2')
    clf, stacking_train, stacking_oot = train_by_cv(X_woe, y, X_oot_woe, y_oot, sss, clf)
    clf.fit(X_woe, y)
    plotcut(clf.predict_proba(X_oot_woe)[:, 1], y_oot, cuts=[10, 20, 30, 40, 50, 60, 70, 80, 90])
