# -*- coding: utf-8 -*-
# @Time    : 2019-01-10 18:16
# @Author  : finupgroup
# @FileName: BoxCoxTransformation.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from scipy import stats,special

class BoxCoxTransformation():
    """
     the seminal paper by Box and Cox(1964)
    """
    def __init__(self,methods,steps=300,isPlot=False):
        self.fillna_method = methods
        self.useful_col = list()
        self.todo_col = list()
        self.fit_result = dict()
        self.steps = steps
        self.isPlot = isPlot


    @staticmethod
    def nan_replace(x,method):
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


    def fit(self,X):
        tmp = X.copy()
        res = pd.DataFrame(tmp.min()).reset_index()
        self.useful_col = list(res[res[0]>=0]['index'])
        self.todo_col = list(res[res[0]<0]['index'])

        for i in self.useful_col:
            tmp[i] = self.nan_replace(tmp[i],method=self.fillna_method)
            tmp[i] = tmp[i].apply(lambda x:1e-10 if x == 0 else x)

            lam_range = np.linspace(-3, 5, self.steps)  # default nums=50
            llf = np.zeros(lam_range.shape, dtype=float)

            # lambda estimate:
            for j, lam in enumerate(lam_range):
                llf[j] = stats.boxcox_llf(lam, tmp[i])  # y 必须>0

            lam_best = lam_range[llf.argmax()]
            self.fit_result[i] = lam_best
            print(i)
            print('Suitable lam is: ', round(lam_best, 2))
            print('Max llf is: ', round(llf.max(), 2))
            print('-------------------------------------------')

            if self.isPlot:
                import matplotlib.pyplot as plt
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                x_most_normal, lmbda_optimal = np.array(tmp[i]), lam_best
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(lam_range, llf, 'b.-')
                ax.axhline(stats.boxcox_llf(lmbda_optimal, tmp[i]), color='r')
                ax.set_xlabel('lambda parameter')
                ax.set_ylabel('Box-Cox log-likelihood' + '|' + str(i))

                locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
                for lmbda, loc in zip([-2, lmbda_optimal, 4], locs):
                    xt = stats.boxcox(tmp[i], lmbda=lmbda)
                    (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
                    ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
                    ax_inset.plot(osm, osr, 'c.', osm, slope * osm + intercept, 'k-')
                    ax_inset.set_xticklabels([])
                    ax_inset.set_yticklabels([])
                    ax_inset.set_title('$\lambda=%1.2f$' % lmbda)

                plt.show()

        return self


    def transform(self,X):
        X_tmp = X.copy()
        result = pd.DataFrame()
        for i, lam_best in zip(self.fit_result.keys(), self.fit_result.values()):
            result[i] = self.nan_replace(X_tmp[i],method=self.fillna_method)
            result[i] = result[i].apply(lambda x:1e-10 if x == 0 else x)
            result[i] = special.boxcox1p(result[i], lam_best)

        return self.nan_replace(pd.concat([result,pd.DataFrame(X_tmp[self.todo_col])],axis=1),method=self.fillna_method)


    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
