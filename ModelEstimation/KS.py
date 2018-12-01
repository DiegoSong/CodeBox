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
        # plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'b--')
        plt.ylim([0, 1])
        # plt.xlim([0, 1])
        plt.ylabel('cumulative population')
        plt.xlabel('scores')
        plt.show()
        # print x
        # print ks_bads
        # print ks_goods
    if return_thre:
        return max_ks, ks_thre
    else:
        return max_ks