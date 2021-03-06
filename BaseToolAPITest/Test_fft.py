"""
@project = BaseToolAPITest
@file = Test_fft
@author = jianwei_hu@aliyun.cm
@create_time = 2020/2/18
"""

import requests
import json
import matplotlib.pyplot as plt


url = r'http://127.0.0.1:5000/signal_tools/fft'

file_data = open("E:/BaseToolAPITest/test_data.txt", 'r')
data_str = file_data.read()
data_json = json.loads(data_str)

headers = {'Content-Type': 'application/json'}

data = {'sample_rate': 2560, 'data_json': data_json}

req = requests.post(url, headers=headers, data=json.dumps(data))

req_json = req.json()
x = req_json['x']
y = req_json['y']

plt.plot(x, y)
plt.show()



