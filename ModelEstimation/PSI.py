import pandas as pd
import numpy as np


def features_psi(actual, expect, columns):

    def _psi_func(_actual, _expect):
        return np.sum((_actual - _expect) * np.log(_actual / _expect))

    psi_list = []
    for col in columns:
        try:
            print(col)
            cuts_actual = pd.qcut(actual[col], 10, labels=None, duplicates='drop', retbins=True)
            cuts_expect = pd.cut(expect[col], cuts_actual[1], labels=None)
            _actual_cnt = cuts_actual[0].reset_index().groupby(col, as_index=False).count()
            _actual_cnt['index'] = _actual_cnt['index'] / cuts_actual[0].count()
            _expect_cnt = cuts_expect.reset_index().groupby(col, as_index=False).count()
            _expect_cnt['index'] = _expect_cnt['index'] / cuts_expect.count()
            psi = _psi_func(_actual_cnt['index'], _expect_cnt['index'])
            psi_list.append(psi)
        except:
            pass

    df = pd.concat([pd.DataFrame(columns, columns=['columns']), pd.DataFrame(psi_list, columns=['psi'])], axis=1)
    return df.sort_values(['psi'], ascending=False)
