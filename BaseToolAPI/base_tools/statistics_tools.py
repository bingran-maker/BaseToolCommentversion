"""
@project = BaseToolAPI
@file = statistics_tools
@author = jianwei_hu@aliyun.cm
@create_time = 2020/2/19
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import timedelta


def ar_ma(data):
    """
    AR自相关分析
    :param data: 时间序列数据，类型：list
    :return:
    {
        'x': [],
        'y': []
    }
    """
    data = np.array(data)
    data = data.astype(np.float64)
    data = pd.Series(data)
    data.index = pd.Index(sm.tsa.datetools.dates_from_range('2000m1', length=len(data)))
    ar_ma_qp70 = sm.tsa.ARMA(data, (7, 1)).fit()
    data_pred_start = pd.date_range(data.index[-1], data.index[-1] + timedelta(weeks=20), freq='1m')[0]
    data_pred_end = pd.date_range(data.index[-1], data.index[-1] + timedelta(weeks=20), freq='1m')[-1]
    data_predict = ar_ma_qp70.predict(start=data_pred_start, end=data_pred_end, dynamic=True)
    print(type(data_predict), data_predict)
    return {'x': data_predict.index.tolist(), 'y': data_predict.values.tolist()}


def outlier(data):
    """
    野点剔除
    :param data: 原始数据
    :return: 剔除野点之后的数据
    """
    data = np.array(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    data_median = np.median(data)
    data_3 = np.percentile(data, 75)
    data_1 = np.percentile(data, 25)
    iqr = data_3 - data_1
    # 用相邻的两个数的均值替换原数据
    for i in np.arange(1, len(data) - 1):
        # i = 1
        if (data[i]>(data_3 + 3 * iqr)) | (data[i] < (data_1 - 3 * iqr)):
            data[i] = (data[i - 1] + data[i + 1]) / 2
    return data.tolist()
