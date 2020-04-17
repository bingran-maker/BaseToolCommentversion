"""
@project = BaseToolAPITest
@file = Test_signal_features
@author = jianwei_hu@aliyun.cm
@create_time = 2020/2/19
"""

import requests
import json
import matplotlib.pyplot as plt


url = r'http://127.0.0.1:5000/signal_tools/signal_features'
file_data = open("E:/BaseToolAPITest/test_data.txt", 'r')
data_str = file_data.read()
data_json = json.loads(data_str)

headers = {'Content-Type': 'application/json'}

data = {'data_json': data_json}

req = requests.post(url, headers=headers, data=json.dumps(data))

req_json = req.json()

print(req_json)