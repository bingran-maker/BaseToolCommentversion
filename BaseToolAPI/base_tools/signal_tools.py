"""
@project = BaseToolAPI
@file = signal_tools
@author = jianwei_hu@aliyun.cm
@create_time = 2020/2/17
"""

import numpy as np
import pandas as pd
import librosa
import statsmodels.api as sm
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal
from scipy.signal import hilbert
from scipy.signal import hanning
from scipy.stats import kurtosis as kur
from scipy.stats import skew as skew
from sklearn.decomposition import PCA
import pywt


def fft(data, sample_rate):
    """
    计算信号数据的FFT
    :param sample_rate: 数据采样频率
    :param data: 原始信号数据
    :return:FFT的 频率和幅值 对应字典
    """
    data_np = np.array(data)
    data_length = len(data)
    data_np = data_np - np.mean(data_np)
    fft_rate = data_length / sample_rate
    fft_data = np.fft.fft(data_np)
    len_data = len(fft_data)
    fft_data = fft_data[range(int(len(fft_data) / 2))]
    fft_data = np.abs(fft_data) / len_data * 2
    fft_data = pd.Series(fft_data)
    fft_data.index = fft_data.index / fft_rate
    x = list(fft_data.index)
    y = list(fft_data.values)
    return {'x': x, 'y': y}


def ana_fft(data, sample_rate):
    """
    快速傅里叶变换
    :param sample_rate: 采样频率
    :param data: 信号数据，类型：list
    :return: fft
    {
        'x':[],
        'y':[]
    }
    """

    data_length = len(data)
    data_rate = data_length / sample_rate
    data_arr = np.array(data)
    data_arr = data_arr - np.mean(data_arr)
    data_fft = np.fft.fft(data_arr)
    data_fft = data_fft[range(len(data_fft) // 2)]
    data_fft = np.abs(data_fft) / len(data_arr) * 2
    data_fft = pd.Series(data_fft)
    data_fft.index = np.arange(1, len(data_fft) + 1) / data_rate
    return {'x': data_fft.index.tolist(), 'y': data_fft.values.tolist()}


def get_peak(fft_amp):
    """
    求得数组前一半中的极值并按大小顺序排列极值
    先求得数组中的极值和极值在数组中的下标,再按顺序排列
    :param fft_amp: #进行FFT后的幅值，类型：list
    :return: dblPeakArray 峰值
    :return: intPeakPosition 峰值位置
    {
        'x': 38,
        'y': 1.232
    }
    """

    fft_np = np.array(fft_amp)
    x = signal.argrelextrema(fft_np, np.greater)[0]
    y = fft_np[signal.argrelextrema(fft_np, np.greater)]

    return {'x': x.tolist(), 'y': y.tolist()}


def envelope(data, sample_rate):
    """
    envelope 包络
    :param sample_rate: 采样频率
    :param data: 信号数据，类型：list
    :return: 包络信号
    {
        'x': [],
        'y': []
    }
    """
    data_length = len(data)
    data_rate = data_length / sample_rate

    data = np.array(data)
    data_h = hilbert(data)
    data_en = np.sqrt(data_h * data_h + data * data)
    data_en = pd.Series(data_en)
    
    data_en.index = np.arange(1, len(data_en) + 1) / data_rate
    data_real = np.real(data_en)
    return {'x': data_en.index.tolist(), 'y': data_real.tolist()}


def butter_filter(data, order, wn, b_type):
    """
    巴特沃斯滤波
    :param data: 信号数据，类型：list
    :param order: 阶数
    :param wn: 临界频率，范围：0~1
    :param b_type:
    :return: 滤波后的信号数据
    {
        'x': [],
        'y': []
    }
    """
    # order = 3
    # wn =0.01  # Wn越小，滤波后越光滑。
    # b_type ='low'
    b, a = signal.butter(N=order, Wn=wn, btype=b_type)
    data_filter = signal.lfilter(b, a, data)
    return data_filter


def stft(data, sample_rate=2560, win_length=1280):
    """
    短时傅里叶变换stft
    :param win_length: 窗宽
    :param sample_rate: 采样频率
    :param data: 信号数据
    :return:[[],[],[]]
    {
        'x': [],
        'y': []
    }
    """
    data_rate = len(data) / sample_rate
    data_np = np.array(data)
    data_stft = librosa.stft(data_np, sample_rate, win_length=win_length)
    data_stft = data_stft.real / len(data) * 2
    data_return = data_stft.tolist()
    return data_stft.tolist()


def cepstrum(data):
    """
    倒谱
    :param data: 信号数据，类型：list
    :return: 进行倒谱转换之后的数据
    {
        'x': [],
        'y': []
    }
    """
    data_length = len(data)
    data = data - np.mean(data)
    x_fft = np.fft.fft(data)
    x_fft = x_fft[:len(x_fft) // 2]
    x_fft_log = np.log(np.abs(x_fft))
    x_fft_log_i_fft = np.fft.ifft(x_fft_log)
    n = len(x_fft_log_i_fft)
    x_fft_log_i_fft_l = []
    for i in range(0, n // 2):
        x_fft_log_i_fft_l.append(x_fft_log_i_fft[n // 2 - i])
    for i in range(n // 2, n):
        x_fft_log_i_fft_l.append(x_fft_log_i_fft[i - n // 2])
    x_cepstrum = pd.Series(x_fft_log_i_fft_l)
    x_cepstrum.index = np.arange(-len(x_cepstrum) // 2, len(x_cepstrum) // 2)
    return {'x': x_cepstrum.index.tolist(), 'y': x_cepstrum.values.real.tolist()}


def simpson(data):
    """
    simpson积分(辛普森积分法)
    :param data: 信号数据
    :return: 积分后的信号
    """
    data_set = np.array(data)
    simpson_result = np.arange(len(data))
    simpson_result = simpson_result.astype(np.float32)
    # 起点与终点 data[-1] = 0 与 data[n] = data[n-1]
    data_set[1:] = data_set[:-1]
    data_set[0] = 0
    # 去均值
    remove_mean_data = data_set - np.mean(data_set)
    simpson_result[0] = 1.0 / 6 * remove_mean_data[0]
    for i in np.arange(len(data) - 1):
        sum_data = 0
        for j in range(1, i + 1):
            sum_data += remove_mean_data[j - 1] + 4 * remove_mean_data[j] + remove_mean_data[j + 1]
        simpson_result[i + 1] = 1.0 / 6 * sum_data
    return simpson_result.tolist()


def signal_features(data):
    """
    时域指标计算
    :param data: 信号数据 data为list类型
    :return: 时域特征指标
    """
    params_keys = ['有效值', '峭度指标', '平均功率', '均值',
                   '绝对均值', '峰值', '方根幅值', '峰峰值',
                   '方差', '标准差', '偏斜度指标', '波形指标',
                   '峰值指标', '脉冲指标', '裕度指标', '偏度值',
                   '峭度值']
    data_arr = np.array(data)
    rms = np.sqrt(np.sum(np.power(data_arr, 2)) / data_arr.shape[0])
    kurtosis = kur(data_arr) + 3
    avg_power = np.sum(np.power(data_arr, 2)) / data_arr.shape[0]
    mean = np.sum(data_arr) / data_arr.shape[0]
    abs_mean = np.sum(np.abs(data_arr)) / data_arr.shape[0]
    sum_peak = 0
    for i in range(8):
        # i = 7
        sum_peak += max(np.abs(data_arr[i * 512:(i + 1) * 512]))
    peak = sum_peak / 8
    root_amp = np.power(np.sum(np.sqrt(np.abs(data_arr))) / data_arr.shape[0], 2)
    p_peak = max(data_arr) - min(data_arr)
    variance = np.sum(np.power(data_arr - mean, 2)) / (data_arr.shape[0] - 1)
    std = np.sqrt(variance)
    skews = skew(data_arr)
    waveform_ind = rms / abs_mean
    crest_ind = max(np.abs(data_arr)) / rms
    impulse_ind = max(np.abs(data_arr)) / abs_mean
    margin_ind = max(np.abs(data_arr)) / root_amp
    skews_ind = skews * np.power(std, 3)
    kur_ind = kurtosis * np.power(std, 4)
    params_values = [rms, kurtosis, avg_power, mean, abs_mean, peak, root_amp, p_peak, variance, std, skews,
                     waveform_ind,
                     crest_ind, impulse_ind, margin_ind, skews_ind, kur_ind]
    params_feats = str(dict(zip(params_keys, params_values)))
    return params_feats
