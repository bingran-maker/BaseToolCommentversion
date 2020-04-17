import librosa
from flask import Flask
from flask import make_response
from flask import request
from functools import wraps
from base_tools import signal_tools, statistics_tools
import logging
import json
from flask import jsonify


app = Flask(__name__)


# 跨域访问装饰器
def allow_cross_domain(fun):
    """
    装饰器 允许app跨域访问
    :param fun: 固定写法
    :return: None
    """
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        rst = make_response(fun(*args, **kwargs))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
        allow_headers = "Referer,Accept,Origin,User-Agent"
        rst.headers['Access-Control-Allow-Headers'] = allow_headers
        return rst
    return wrapper_fun


@app.route('/')
@allow_cross_domain
def hello_world():
    return 'Hello World!'


@app.route('/signal_tools/fft', methods=['POST'])
@allow_cross_domain
def fft():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            signal_data = data_json['data_json']
            sample_rate = data_json['sample_rate']
            fft_data = signal_tools.fft(signal_data, sample_rate)
            return json.dumps(fft_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/ana_fft', methods=['POST'])
@allow_cross_domain
def ana_fft():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            signal_data = data_json['data_json']
            sample_rate = data_json['sample_rate']
            fft_data = signal_tools.ana_fft(signal_data, sample_rate)
            return json.dumps(fft_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/get_peak', methods=['POST'])
@allow_cross_domain
def get_peak():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            fft_data = data_json['data_json']
            peak_data = signal_tools.get_peak(fft_data)
            return json.dumps(peak_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/envelope', methods=['POST'])
@allow_cross_domain
def envelope():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            sample_rate = data_json['sample_rate']
            envelope_data = signal_tools.envelope(raw_data, sample_rate)
            print(envelope_data)
            return json.dumps(envelope_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/stft', methods=['POST'])
@allow_cross_domain
def stft():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            sample_rate = data_json['sample_rate']
            win_length = data_json['win_length']
            print(win_length, sample_rate, raw_data)
            stft_data = signal_tools.stft(data=raw_data, sample_rate=sample_rate, win_length=win_length)
            print(type(stft_data), stft_data)
            return json.dumps(stft_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/butter_filter', methods=['POST'])
@allow_cross_domain
def butter_filter():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            wn = data_json['wn']
            order = data_json['order']
            # b_type = "low"  "high"
            b_type = data_json['b_type']
            if b_type is None or b_type == '':
                b_type = 'low'
            print(wn, order, b_type)
            butter_data = signal_tools.butter_filter(raw_data, order=order, wn=wn, b_type=b_type)
            print(type(butter_data), butter_data)
            return json.dumps(butter_data.tolist())
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/cepstrum', methods=['POST'])
@allow_cross_domain
def cepstrum():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            cepstrum_data = signal_tools.cepstrum(raw_data)
            print(type(cepstrum_data), cepstrum_data)
            return json.dumps(cepstrum_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/simpson', methods=['POST'])
@allow_cross_domain
def simpson():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            simpson_data = signal_tools.simpson(raw_data)
            print(type(simpson_data), simpson_data)
            return json.dumps(simpson_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/signal_tools/signal_features', methods=['POST'])
@allow_cross_domain
def signal_features():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            features_data = signal_tools.signal_features(raw_data)
            return json.dumps(features_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/statistics_tools/ar_ma', methods=['POST'])
@allow_cross_domain
def ar_ma():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            ar_ma_data = statistics_tools.ar_ma(raw_data)

            return json.dumps(ar_ma_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


@app.route('/statistics_tools/outlier', methods=['POST'])
@allow_cross_domain
def outlier():
    if request.method == 'POST':
        try:
            raw_data = request.get_data()
            data_json = json.loads(raw_data)
            raw_data = data_json['data_json']
            outlier_data = statistics_tools.outlier(raw_data)

            return json.dumps(outlier_data)
        except Exception as e:
            return {"error": e}
    else:
        return {"error": "请使用POST方法调用API接口"}


if __name__ == '__main__':
    # app.run(host='0.0.0.0',port=5000, threaded=True)
    app.run(host='127.0.0.1', port=5000, threaded=True)
