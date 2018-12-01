from lightgbm import LGBMRegressor


def fillna_by_model(opt, data, cols):
    df = data[cols].copy()
    for col in cols:
        X_train = df[df[col].notnull()].drop(columns=col)
        X_test = df[df[col].isnull()].drop(columns=col)
        if X_test.shape[0] == 0:
            continue
        y_train = df[df[col].notnull()][col]
        opt.fit(X_train, y_train)
        y_test = opt.predict(X_test)
        data.loc[data[col].isnull(), col] = y_test
        print("%s Done." % col)
    return data


reg = LGBMRegressor()
cols = ['credit_limit_rmb', 'new_balance_rmb', 'cash_advance_limit_rmb', 'min_payment_rmb']
data = fillna_by_model(reg, data, cols)
data.head()
