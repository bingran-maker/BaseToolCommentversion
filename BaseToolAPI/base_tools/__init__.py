"""
@project = BaseToolAPI
@file = __init__
@author = jianwei_hu@aliyun.cm
@create_time = 2020/2/17
"""

import os
import sys

pro_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pro_path)
for root, dirs, files in os.walk(pro_path):
    for file in files:
        name, ext = os.path.splitext(file)
        if ext == '.py' and name != '__init__' and pro_path == root:
            __import__(name)

        for dir in dirs:
            if dir != '.svn':
                try:
                    __import__(__name__ + '.' + dir)
                except:
                    pass
