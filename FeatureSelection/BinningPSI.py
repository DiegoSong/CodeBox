#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

class TableBinPsi(object):
    def __int__(self):
        self.table_calcule = pd.DataFrame()
        self.nbins_cats = 10
        self.num_to_char_levels = 50
        self.varlist = []
        self.nbins = 10
        self.var_list_psi = []
        self.label = ''

    def fit(self, train, test, varlist, num_to_char_levels, nbins, nbins_cats):
        '''
        :param data: pd.DataFrame()
        :param varlist: list of variables
        :num_to_char_levels: maximum levels for characters
        :nbins: number of bins
        :nbins_cats: number of category bins
        :return: binning result of variables
        '''
        if len(varlist) > 0:
            train = train[varlist].copy()
            test = test[varlist].copy()
        
        train_continues = []
        train_category = []
        for c in varlist:
            try:
                train[c] = train[c].astype('float')
                train_continues.append(c)
            except ValueError:
                train_category.append(c)
                
#         train_continues = list(train.select_dtypes(exclude=['O']).columns)
#         train_category = list(train.select_dtypes(include=['O']).columns)
        conti_add = []
        for eltchar in train_continues:
            if (len(pd.DataFrame(pd.value_counts(train[eltchar])))) < num_to_char_levels:
                conti_add.append(eltchar)
        if len(conti_add) > 0:
            for ap in conti_add:
                train_category.append(ap)
                train_continues.remove(ap)
        psi_result_continue = pd.DataFrame()
        for elt1 in train_continues:
            #print(elt1)
            data = self.variable_continue(train, test, elt1, nbins)
            psi_result_continue = pd.concat([psi_result_continue, data], axis=0, ignore_index=True)
        psi_result_description = pd.DataFrame()
        for elt1 in train_category:
            #print(elt1)
            data = self.variable_desc(train, test, elt1, nbins_cats)
            psi_result_description = pd.concat([psi_result_description, data], axis=0, ignore_index=True)
        #print(psi_result_continue.head())
        #print(psi_result_description.head())
        table_bins = pd.concat([psi_result_description, psi_result_continue], axis=0, ignore_index=True)
        self.table_calcule = table_bins.copy()
        return table_bins

    def transform(self,var_list_psi):
        '''
        :param var_list_psi: list of calcule variables
        :return: psi_result:table of variables psi
        '''
        data = self.table_calcule
        psi_result = pd.DataFrame()
        for elt in var_list_psi:
            analyse_data = data.loc[data['varnames'] == elt, :].copy()
            analyse_data.reset_index(drop=False)
#            analyse_data.loc[(analyse_data['oos_percent'] == 0) & (analyse_data['ins_percent'] <= 0.001), 'ins_percent'] = 1
            analyse_data.loc[(analyse_data['oos_percent'] == 0), 'oos_percent'] = 0.001
            analyse_data.loc[(analyse_data['ins_percent'] == 0), 'ins_percent'] = 0.001
            partb = list(analyse_data['oos_percent']/analyse_data['ins_percent'])
            logpartb = []
            for i in range(len(partb)):
                logpartb.append(math.log(partb[i]))
            parta = list(analyse_data['oos_percent']-analyse_data['ins_percent'])
            psi = 0
            for j in range(len(parta)):
                psi += parta[j] * logpartb[j]
            psi_prog = pd.DataFrame({'varnames': elt, 'psi': psi}, index=[0],columns=['varnames', 'psi'])
            psi_result = pd.concat([psi_result, psi_prog], axis=0)
        return psi_result.reset_index(drop=True).sort_values('psi', ascending=False)

    def cal_informationvalue(self, train, test, label):
        '''

        :param train: same as fit
        :param test: same as fit
        :param label: 'is_overdue'
        :return: InformationValue DataFrame
        '''
        matrix = self.table_calcule
        ins = train[~train[label].isnull()].reset_index(drop=True)
        oos = test[~test[label].isnull()].reset_index(drop=True)

        matrix['ins_woe1'] = 0
        matrix['oos_woe1'] = 0
        cat_list = list(set([x for x in matrix[matrix['bins_min'] == 'categorical']['varnames']]))
        num_list = list(set([x for x in matrix[matrix['bins_min'] != 'categorical']['varnames']]))
        for elt in cat_list:
            tmp = list(set(matrix[matrix['varnames'] == elt]['bins_numbers']) - set(['others']))
            for i in matrix[matrix['varnames'] == elt]['bins_numbers']:
                if i != 'others':
                    if (len(ins[ins[elt] == i][label]) - sum(ins[ins[elt] == i][label])) == 0 or (len(oos[oos[elt] == i][label]) - sum(oos[oos[elt] == i][label])) == 0 or (ins.shape[0] - sum(ins[label])) == 0 or (oos.shape[0] - sum(oos[label])) == 0:
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_iv1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_iv1'] = np.nan
                    else:
                        a = sum(ins[ins[elt] == i][label])/(len(ins[ins[elt] == i][label]) - sum(ins[ins[elt] == i][label]))
                        b = sum(ins[label])/(ins.shape[0] - sum(ins[label]))

                        c = sum(oos[oos[elt] == i][label])/(len(oos[oos[elt] == i][label]) - sum(oos[oos[elt] == i][label]))
                        d = sum(oos[label])/(oos.shape[0] - sum(oos[label]))

                        if a/b == 0:
                            woe1 = np.nan
                        else:
                            woe1 = np.log(a / b)
                        if c/d == 0:
                            woe2 = np.nan
                        else:
                            woe2 = np.log(c / d)

                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_woe1'] = woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_iv1'] = (a-b) * woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_woe1'] = woe2
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_iv1'] = (c-d) * woe2
                else:
                    tt = ins[~ins[elt].isin(tmp)].reset_index(drop=True)
                    mm = oos[~oos[elt].isin(tmp)].reset_index(drop=True)
                    if (len(tt[label]) - sum(tt[label])) == 0 or (ins.shape[0] - sum(ins[label])) == 0 or (len(mm[label]) - sum(mm[label])) == 0 or (oos.shape[0] - sum(oos[label])) == 0:
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_iv1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_iv1'] = np.nan
                    else:
                        a = sum(tt[label]) / (len(tt[label]) - sum(tt[label]))
                        b = sum(ins[label]) / (ins.shape[0] - sum(ins[label]))

                        c = sum(mm[label])/(len(mm[label]) - sum(mm[label]))
                        d = sum(oos[label])/(oos.shape[0] - sum(oos[label]))

                        if a/b == 0:
                            woe1 = np.nan
                        else:
                            woe1 = np.log(a / b)
                        if c/d == 0:
                            woe2 = np.nan
                        else:
                            woe2 = np.log(c / d)

                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_woe1'] = woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'ins_iv1'] = (a-b) * woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_woe1'] = woe2
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_numbers'] == i), 'oos_iv1'] = (c-d) * woe2
        for elt in num_list:
            for i,j in zip(matrix[matrix['varnames'] == elt]['bins_min'], matrix[matrix['varnames'] == elt]['bins_max']):
                if i != 'NONE':
                    if (len(ins[(ins[elt] > i) & (ins[elt] <= j)][label]) - sum(ins[(ins[elt] > i) & (ins[elt] <= j)][label])) == 0 or (ins.shape[0] - sum(ins[label])) == 0 or (len(oos[(oos[elt] > i) & (oos[elt] <= j)][label]) - sum(oos[(oos[elt] > i) & (oos[elt] <= j)][label])) == 0 or (oos.shape[0] - sum(oos[label])) == 0:
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_iv1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_iv1'] = np.nan
                    else:
                        a = sum(ins[(ins[elt] > i) & (ins[elt] <= j)][label])/(len(ins[(ins[elt] > i) & (ins[elt] <= j)][label]) - sum(ins[(ins[elt] > i) & (ins[elt] <= j)][label]))
                        b = sum(ins[label])/(ins.shape[0] - sum(ins[label]))

                        c = sum(oos[(oos[elt] > i) & (oos[elt] <= j)][label])/(len(oos[(oos[elt] > i) & (oos[elt] <= j)][label]) - sum(oos[(oos[elt] > i) & (oos[elt] <= j)][label]))
                        d = sum(oos[label])/(oos.shape[0] - sum(oos[label]))

                        if a/b == 0:
                            woe1 = np.nan
                        else:
                            woe1 = np.log(a / b)
                        if c/d == 0:
                            woe2 = np.nan
                        else:
                            woe2 = np.log(c / d)

                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_woe1'] = woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_iv1'] = (a-b) * woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_woe1'] = woe2
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_iv1'] = (c-d) * woe2
                else:
                    if (len(ins[ins[elt].isnull()][label]) - sum(ins[ins[elt].isnull()][label])) == 0 or (ins.shape[0] - sum(ins[label])) == 0 or (len(oos[oos[elt].isnull()][label]) - sum(oos[oos[elt].isnull()][label])) == 0 or (oos.shape[0] - sum(oos[label])) == 0:
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_iv1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_woe1'] = np.nan
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_iv1'] = np.nan
                    else:
                        a = sum(ins[ins[elt].isnull()][label]) / (len(ins[ins[elt].isnull()][label]) - sum(ins[ins[elt].isnull()][label]))
                        b = sum(ins[label]) / (ins.shape[0] - sum(ins[label]))

                        c = sum(oos[oos[elt].isnull()][label]) / (len(oos[oos[elt].isnull()][label]) - sum(oos[oos[elt].isnull()][label]))
                        d = sum(oos[label]) / (oos.shape[0] - sum(oos[label]))

                        if a/b == 0:
                            woe1 = np.nan
                        else:
                            woe1 = np.log(a / b)
                        if c/d == 0:
                            woe2 = np.nan
                        else:
                            woe2 = np.log(c / d)

                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_woe1'] = woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'ins_iv1'] = (a-b) * woe1
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_woe1'] = woe2
                        matrix.loc[(matrix['varnames'] == elt) & (matrix['bins_min'] == i), 'oos_iv1'] = (c-d) * woe2

        IV_result = pd.DataFrame()
        for elt in cat_list + num_list:
            analyse_data = matrix.loc[matrix['varnames'] == elt, :].copy()
            analyse_data.reset_index(drop=False)

            ins_IV = analyse_data.groupby(['varnames']).agg({'ins_iv1':['sum'],'oos_iv1':['sum']}).values[0][0]
            oos_IV = analyse_data.groupby(['varnames']).agg({'ins_iv1': ['sum'], 'oos_iv1': ['sum']}).values[0][1]

            IV_prog = pd.DataFrame({'varnames': elt, 'ins_InformationValue': ins_IV, 'oos_InformationValue': oos_IV}, index=[0], columns=['varnames', 'ins_InformationValue', 'oos_InformationValue'])
            IV_result = pd.concat([IV_result, IV_prog], axis=0)

        return IV_result

    @staticmethod
    def variable_continue(ins, oos, elt, nbins):
        '''
        calculate table of continuous variables
        :param ins: train data
        :param oos: test data
        :param elt: list of continuous variables
        :param nbins: numbers of bin cuts
        :return:result_continuous
        '''
        # select colomn elt
        elt_null_ins = pd.isnull(ins[elt])
        elt_null_true_ins = ins[elt][elt_null_ins == True]
        if len(elt_null_true_ins) > 0:
            elt_null_true_ins = ["NONE"] * len(elt_null_true_ins)
        else:
            elt_null_true_ins = ["NONE"]

        elt_null_oos = pd.isnull(oos[elt])
        elt_null_true_oos = oos[elt][elt_null_oos == True]
        if len(elt_null_true_oos) > 0:
            elt_null_true_oos = ["NONE"] * len(elt_null_true_oos)
        else:
            elt_null_true_oos = ["NONE"]

        elt_null_false_ins = ins[elt][elt_null_ins == False]
        elt_null_false_oos = oos[elt][elt_null_oos == False]

        df_base_null = pd.DataFrame(elt_null_true_ins).copy()
        df_base_null = df_base_null.reset_index(drop=True)
        df_oot_null = pd.DataFrame(elt_null_true_oos).copy()
        df_oot_null = df_oot_null.reset_index(drop=True)

        df_base = pd.DataFrame(elt_null_false_ins).copy()
        df_base = df_base.reset_index(drop=True)
        df_oot = pd.DataFrame(elt_null_false_oos).copy()
        df_oot = df_oot.reset_index(drop=True)

        # nbins binning
        ind = range(0, df_base.shape[0] + 1)
        a, ind_index = pd.qcut(ind, nbins - 1, retbins=True)
        ind_index = ([int(x) for x in ind_index])

        # get nbins cut points
        # sort by elt
        sort_elt = df_base.copy()
        sort_elt['sort_elt'] = sorted(sort_elt[elt], reverse=False)
        cut_points = sort_elt.loc[ind_index, ['sort_elt']]
        cut_points = np.array(cut_points[1:(len(cut_points) - 1)]['sort_elt'])
        cut_points = list(set(cut_points.tolist()))
        cut_points.append(-np.inf)
        cut_points.append(np.inf)
        # duplicated cut points
        cut_points = sorted(cut_points)
        bin_base = pd.cut(pd.Series(df_base[elt]), cut_points, include_lowest=True)
        try:
            bin_oot = pd.cut(pd.Series(df_oot[elt]), cut_points, include_lowest=True)
        except TypeError:
            bin_oot = []

        ins_total = pd.concat([bin_base, df_base_null], axis=0, ignore_index=True)
        oos_total = pd.concat([bin_oot, df_oot_null], axis=0, ignore_index=True)
        ins_total.columns = [elt]
        oos_total.columns = [elt]

        if len(bin_oot) > 0:
            ins_total['bins'] = ins_total
            oos_total['bins'] = oos_total
            # result analysis
            grouped_base = ins_total.groupby(['bins']).agg({elt: ['count']}).rename(columns={elt:'count_ins'})
            grouped_base['bins'] = grouped_base.index
            # print(grouped_base)
            grouped_oot = oos_total.groupby(['bins']).agg({elt: ['count']}).rename(columns={elt:'count_oos'})
            grouped_oot['bins'] = grouped_oot.index
            # print(grouped_oot)
            grouped_oot = grouped_oot.reset_index(drop=True)
            grouped_base = grouped_base.reset_index(drop=True)
            result = pd.merge(grouped_base, grouped_oot, how='left', on=['bins'])
            result = result.fillna(0)
            min_col = cut_points[0:(len(cut_points))-1]
            min_col.append('NONE')
            max_col = cut_points[1:(len(cut_points))]
            max_col.append('NONE')

            if len(result['bins']) != len(min_col):
                _tmp_last = result.iloc[-1, :]
                result.iloc[-1, :] = [0, pd.Interval(min_col[-2], np.inf), 0]
                result = result.append(_tmp_last).reset_index(drop=True)

            result['bins_min'] = min_col
            result['bins_max'] = max_col
            if result['count_oos'].loc[result.shape[0]-1][0]==1:
                result.loc[result.shape[0]-1,'count_oos'] = 0.0
            if result['count_ins'].loc[result.shape[0]-1][0]==1:
                result.loc[result.shape[0]-1,'count_ins'] = 0.0
            result['ins_percent'] = result['count_ins'] * 1.0 / (ins_total.shape[0]-1)
            result['oos_percent'] = result['count_oos'] * 1.0 / (oos_total.shape[0]-1)
            result['varnames'] = [elt] * result.shape[0]
            result['bins_numbers'] = range(1,result.shape[0]+1,1)
            resetnames = ['varnames','bins_numbers','bins_min','bins_max','count_ins','ins_percent','count_oos','oos_percent']
            result_continuous = pd.DataFrame()
            for col in resetnames:
                result_continuous[col] = result[col]
        else:
            result_continuous = []
            print(elt + ' feature error !!!')
        return result_continuous

    @staticmethod
    def variable_desc(ins, oos, elt, nbins_cats):
        '''
        calculate table of discretes variables
        :param ins: train data
        :param oos: test data
        :param elt: list of discretes variables
        :return: result_discretes
        '''
        # select colomn elt
        #print(elt)
        df_base = pd.DataFrame(ins[elt])
        df_base = df_base.fillna('others')
        df_oot = pd.DataFrame(oos[elt])
        df_oot = df_oot.fillna('others')

        rename_less_cat = pd.DataFrame(pd.value_counts(df_base[elt]))
        if rename_less_cat.shape[0] > nbins_cats:
            for rnum in range((nbins_cats - 1), len(rename_less_cat)):
                rena = rename_less_cat.index[rnum]
                df_base[elt] = df_base[elt].replace(rena, 'others')
        else:
            rena = rename_less_cat.index[len(rename_less_cat) - 1]
            df_base[elt] = df_base[elt].replace(rena, 'others')

        base_levels = set(pd.unique(df_base[elt]))
        oot_levels = set(pd.unique(df_oot[elt]))
        repl01 = list(base_levels - oot_levels)
        if len(repl01) > 0:
            for i in repl01:
                df_base[elt] = df_base[elt].replace(i, 'others')
        repl02 = list(oot_levels - base_levels)
        if len(repl02) > 0:
            for i in repl02:
                df_oot[elt] = df_oot[elt].replace(i, 'others')
        grouped_base = pd.DataFrame(pd.value_counts(df_base[elt]))
        grouped_oot = pd.DataFrame(pd.value_counts(df_oot[elt]))
        grouped_base.columns = ['count_ins']
        grouped_oot.columns = ['count_oos']
        grouped_base['bins'] = grouped_base.index
        grouped_oot['bins'] = grouped_oot.index

        result = pd.merge(grouped_base, grouped_oot, how='left', on=['bins'])
        result = result.fillna(0)
        result['bins_min'] = ['categorical']*result.shape[0]
        result['bins_max'] = ['categorical']*result.shape[0]
        result['ins_percent'] = result['count_ins'] * 1.0 / (df_base.shape[0])
        result['oos_percent'] = result['count_oos'] * 1.0 / (df_oot.shape[0])
        result['varnames'] = [elt] * result.shape[0]
        result['bins_numbers'] = result['bins']
        resetnames = ['varnames', 'bins_numbers', 'bins_min', 'bins_max', 'count_ins', 'ins_percent', 'count_oos',
                      'oos_percent']
        result_discretes = result[resetnames].copy()
        return result_discretes


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    pd.set_option("max_columns", 100)
    import warnings

    warnings.filterwarnings('ignore')

    raw = pd.read_csv("/Users/finup/Project/MoonLight/data/v20190311/t.csv")

    id_features = ['order_id', 'phone', 'inner_sources', 'create_time', 'plan_repay_date', 'is_overdue']
    xj_model_col = ['age', 'gender', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum',
                    'history_meal_fee_mean_180_min', 'gsm_active_afternoon_hours_ratio_180_min',
                    'gsm_active_night_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_sum',
                    'gsm_active_morning_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_std',
                    'gsm_active_night_hours_ratio_180_max', 'sms_active_morning_hours_ratio_180_mean',
                    'gsm_communication_way_dail_180_std', 'sms_communication_way_received_180_skew',
                    'gsm_active_evening_hours_ratio_180_max', 'gsm_communication_way_dailed_180_skew',
                    'sms_active_night_hours_ratio_180_max', 'history_meal_fee_mean_180_mean',
                    'sms_active_evening_hours_ratio_180_max', 'gsm_active_afternoon_hours_180_mean',
                    'gsm_active_afternoon_hours_ratio_180_sum', 'sms_communication_way_received_180_std',
                    'gsm_active_times_180_std', 'gsm_active_afternoon_hours_180_skew', 'sms_active_times_180_skew',
                    'sms_active_morning_hours_ratio_180_sum', 'sms_communication_way_send_180_skew',
                    'gsm_active_night_hours_180_sum']
    ff_model_col = ['age', 'gender', 'op_lasts', 'account_balance', 'gsm_active_night_hours_ratio_180_sum',
                    'history_meal_fee_mean_180_min', 'gsm_active_afternoon_hours_ratio_180_min',
                    'gsm_active_night_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_sum',
                    'gsm_active_morning_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_std',
                    'gsm_active_night_hours_ratio_180_max', 'sms_active_morning_hours_ratio_180_mean',
                    'gsm_communication_way_dail_180_std', 'gsm_active_evening_hours_ratio_180_max',
                    'sms_active_night_hours_ratio_180_max', 'history_meal_fee_mean_180_mean',
                    'sms_active_evening_hours_ratio_180_max', 'gsm_active_afternoon_hours_180_mean',
                    'gsm_active_afternoon_hours_ratio_180_sum', 'sms_communication_way_received_180_std',
                    'gsm_active_times_180_std', 'sms_active_morning_hours_ratio_180_sum',
                    'history_other_fee_180_min',
                    'gsm_active_night_hours_180_sum']
    markcode_col = ['loan_app_nums', 'device_related_id_nums_30', 'device_related_mobile_nums_30',
                    'mobile_related_id_nums_30', 'mobile_related_device_nums_30', 'mobile_qz_current_overdue',
                    'mobile_qz_amount_not_clear', 'mobile_qz_history_overdue_nums',
                    'mobile_superloan_current_overdue',
                    'mobile_superloan_amount_not_clear', 'mobile_superloan_history_overdue_nums',
                    'mobile_bestbuy_current_overdue', 'mobile_bestbuy_amount_not_clear',
                    'mobile_bestbuy_history_overdue_nums', 'mobile_car_finance_current_overdue',
                    'mobile_car_finance_amount_not_clear', 'mobile_car_finance_history_overdue_nums',
                    'mobile_fan_card_current_overdue', 'mobile_fan_card_amount_not_clear',
                    'mobile_fan_card_history_overdue_nums', 'mobile_finup_lend_current_overdue',
                    'mobile_finup_lend_amount_not_clear', 'mobile_finup_lend_history_overdue_nums',
                    'id_related_device_nums_30', 'id_related_mobile_nums_30', 'id_qz_current_overdue',
                    'id_qz_amount_not_clear', 'id_qz_history_overdue_nums', 'id_superloan_amount_not_clear',
                    'id_superloan_history_overdue_nums', 'id_bestbuy_current_overdue',
                    'id_bestbuy_amount_not_clear',
                    'id_bestbuy_history_overdue_nums', 'id_car_finance_current_overdue',
                    'id_car_finance_amount_not_clear', 'id_car_finance_history_overdue_nums',
                    'id_fan_card_current_overdue', 'id_fan_card_amount_not_clear',
                    'id_fan_card_history_overdue_nums',
                    'id_finup_lend_current_overdue', 'id_finup_lend_amount_not_clear',
                    'id_finup_lend_history_overdue_nums', 'phonebook_contacts_nums',
                    'phonebook_finup_registered_ratio',
                    'phonebook_finup_history_overdue_nums', 'phonebook_finup_current_overdue_nums',
                    'emergency_finup_current_overdue_nums', 'emergency_finup_history_overdue_nums',
                    'operator_account_days', 'dailed_record_nums_30']
    sanfang = ['black_overduegrade', 'um_credit_score', 'app_stability_7d', 'loan_7d', 'finance_7d', 'tcrisk_score',
               'rule_final_weight', 'apply_latest_one_month', 'behavior_loans_score']
    important_features = ['age', 'gender', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum', 'black_overduegrade',
                          'um_credit_score', 'loan_7d', 'apply_latest_one_month',
                          'operator_dailed_contacts_top20_30_finup_registered_nums', 'short_dail_ratio_30']

    df = raw[id_features + important_features]
    old = df[(df['plan_repay_date'] >= '2018-01-27') & (df['plan_repay_date'] < '2019-03-05')].reset_index(
        drop=True)
    new = df[(df['plan_repay_date'] >= '2019-03-05') & (df['plan_repay_date'] < '2019-03-07')].reset_index(
        drop=True)
    print(old.shape, new.shape)

    tbp = TableBinPsi()
    table_bins_cross = tbp.fit(old, new, ['age'], 15, 5, 5)

if __name__ == '__main__':
    from src.monitor.monitor_v1 import features_split_time
    raw = pd.read_csv("/Users/finup/Project/MoonLight/data/v20190311/t.csv")
    important_features = ['age', 'gender', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum', 'black_overduegrade',
                          'um_credit_score', 'loan_7d', 'apply_latest_one_month',
                          'operator_dailed_contacts_top20_30_finup_registered_nums', 'short_dail_ratio_30',
                          'behavior_loans_settle_count', 'behavior_history_fail_fee',
                          'current_consfin_org_count', 'behavior_latest_one_month_suc', 'rule_final_weight',
                          'tcrisk_score', 'current_consfin_max_limit', 'apply_latest_six_month',
                          'behavior_loans_overdue_count', 'behavior_history_suc_fee']
    old_period = '2019-01-27,2019-03-05'
    new_period = '2019-03-11,2019-03-14'
    old_plan_repay_date = '2018-01-27'
    new_plan_repay_date = '2019-03-05'
    old_steps = -1
    new_steps = -1
    td = 3
    abnormal_threshold = 0.2

    import_col = ['create_time'] + important_features + ['ipprovince', 'gps_province']
    import_old_time = old_period.split(",")
    import_current_time = new_period.split(",")
    import_old_time_space = str(old_period)
    import_current_time_space = str(new_period)
    import_data_raw = raw[import_col]

    import_data = pd.DataFrame()
    import_data = pd.concat([import_data, features_split_time(import_data_raw, import_old_time, '-1', 'old')])
    import_data = pd.concat([import_data, features_split_time(import_data_raw, import_current_time, '-1', 'current')])

    import_data['ipprovince'] = import_data['ipprovince'].fillna('未知')
    import_data['gps_province'] = import_data['gps_province'].fillna('未知')

    old_import_data = import_data[import_data['time_flag'] == 'old'].reset_index(drop=1)
    current_import_data = import_data[import_data['time_flag'] == 'current'].reset_index(drop=1)

    tbp = TableBinPsi()
    table_bins_cross = tbp.fit(old_import_data, current_import_data, ['tcrisk_score'], 15, 10, 10)