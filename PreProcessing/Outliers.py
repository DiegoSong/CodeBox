
def deal_with_outlier(pdf, cols, n_times=3):
    for col in cols:
        mean = pdf[col].mean()
        std = pdf[col].std()
        pdf.loc[pdf[col] > mean + n_times * std, col] = mean
        pdf.loc[pdf[col] < mean - n_times * std, col] = mean
    return pdf