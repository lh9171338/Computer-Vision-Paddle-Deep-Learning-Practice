# -*- encoding: utf-8 -*-
"""
@File    :   builder.py
@Time    :   2023/09/09 11:34:14
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import paddle
from . import LeNet


def build_model(cfg):
    """
    根据给定的配置构建模型

    Args:
        cfg (Union[dict, Any]): 模型配置参数。

    Returns:
        Any: 根据配置构建完成的模型。
    """
    if not isinstance(cfg, dict):
        cfg = dict(cfg)
    assert cfg['type']  in ['LeNet'], 'Unrecognized model name'
    if cfg['type'] == 'LeNet':
        model = LeNet(**cfg)

    return model


def build_optimizer(cfg, model, scheduler=None):
    """
    根据给定配置构建优化器。

    Args:
        cfg (dict or object): 优化器配置。
        model (paddle.nn.Layer): 待优化模型的参数。
        scheduler (paddle.optimizer.lr.LRScheduler, optional): 学习率调度器。默认为None。

    Returns:
        paddle.optimizer.Optimizer: 优化器实例。
    """
    if not isinstance(cfg, dict):
        cfg = dict(cfg)

    learning_rate = scheduler if scheduler is not None else cfg['learning_rate']
    parameters = model.parameters()
    assert cfg['type'] in ['SGD', 'Adam', 'AdamW'], 'Unrecognized optimizer name'
    if cfg['type'] == 'SGD':
        optimizer = paddle.optimizer.SGD(parameters=parameters,
            learning_rate=learning_rate, weight_decay=cfg['weight_decay'])
    elif cfg['type'] == 'Adam':
        optimizer = paddle.optimizer.Adam(parameters=parameters,
            learning_rate=learning_rate, weight_decay=cfg['weight_decay'])
    elif cfg['type'] == 'AdamW':
        optimizer = paddle.optimizer.AdamW(parameters=parameters,
            learning_rate=learning_rate, weight_decay=cfg['weight_decay'])

    return optimizer


def build_scheduler(cfg):
    """
    根据配置字典构建学习率调度器。

    Args:
        cfg (dict): 学习率调度器的配置。

    Returns:
        paddle.optimizer.lr.StepDecay or paddle.optimizer.lr.CosineAnnealingDecay: 根据配置字典构建的学习率调度器。
    """
    if not isinstance(cfg, dict):
        cfg = dict(cfg)

    assert cfg['type'] in ['StepDecay', 'CosineAnnealingDecay'], 'Unrecognized scheduler name'
    if cfg['type'] == 'StepDecay':
        scheduler = paddle.optimizer.lr.StepDecay(learning_rate=cfg['learning_rate'],
            step_size=cfg['step_size'], gamma=cfg['gamma'])
    elif cfg['type'] == 'CosineAnnealingDecay':
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cfg['learning_rate'],
            T_max=cfg['num_epochs'])

    return scheduler
