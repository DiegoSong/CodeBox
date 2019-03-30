def check_dtype(tdf, varlist, var_diff_num):
    df = tdf.copy()
    cate_f = []
    con_f = []
    con_copy = []
    for a in varlist:
        try:
            df[a] = df[a].astype('float')
            con_f.append(a)
        except ValueError:
            cate_f.append(a)
    for a in con_f:
        if df[a].nunique() <= var_diff_num:
            cate_f.append(a)
        else:
            con_copy.append(a)
    con_f = con_copy
    return cate_f, con_f