# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Time    :   2023/09/09 17:55:13
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize
import logging
import tqdm
from network import build_model
from configs import config


def test(cfg):
    """test"""
    # Load dataset
    transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
    dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
    loader = paddle.io.DataLoader(dataset=dataset, batch_size=cfg.test_batch_size, num_workers=cfg.num_workers, shuffle=False)

    # Load model
    model = build_model(cfg.model)
    state_dict = paddle.load(cfg.model_file)
    model.load_dict(state_dict['model'])

    # Test
    model.eval()

    tp = 0
    for images, labels in tqdm.tqdm(loader, desc='test: '):
        outputs = model(images)
        preds = paddle.argmax(outputs, axis=-1, keepdim=True)
        tp += (preds == labels).sum().item()

    accuracy = float(tp) / float(len(loader.dataset))
    logging.info(f'accuracy: {accuracy:.3f}')


if __name__ == '__main__':
    # Parameter
    cfg = config.parse()

    # set base logging config
    fmt = '[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    logging.info(cfg)

    # Test network
    test(cfg)
