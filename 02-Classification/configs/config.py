# -*- encoding: utf-8 -*-
"""
@File    :   coufig.py
@Time    :   2023/09/09 11:16:35
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--save_path', type=str, default='output', help='save path')
    parser.add_argument('-c', '--config_file', type=str, help='config file')
    parser.add_argument('-m', '--model_file', type=str, help='model file')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate')
    parser.add_argument('--train_batch_size', type=int, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, help='test batch size')

    opts = parser.parse_args()
    opts_dict = vars(opts)
    opts_list = []
    for key, value in opts_dict.items():
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)

    yaml_file = os.path.join(opts.config_file)
    cfg = CfgNode(new_allowed=True)
    cfg.merge_from_file(yaml_file)
    cfg.merge_from_list(opts_list)
    cfg.freeze()

    return cfg
