# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/8/20 21:14
 @desc:
"""

import json


name_dict = {}
dir_dict = {}
idx_dict = {}

with open('data/imagenet_class_index.json', 'r') as json_file:
    class_info = json.load(json_file)  # dict like {"0": ["n01440764", "tench"], ...}
    for k, v in class_info.items():
        name_dict[int(k)] = v[1]
        dir_dict[int(k)] = v[0]
        idx_dict[v[0]] = int(k)

# print(name_dict)
# print(dir_dict)
# print(idx_dict)

