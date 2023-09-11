# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2023/09/09 11:16:59
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize
from visualdl import LogWriter
import random
import numpy as np
import logging
import shutil
import tqdm
from network import build_model, build_scheduler, build_optimizer
from configs import config
from utils import Timer


def format_msg(msg_dict):
    """
    将字典格式的消息转换成字符串格式

    Args:
        msg_dict (dict): 包含消息的字典，键值对格式为 {str: any}

    Returns:
        str: 格式化后的消息字符串

    """
    for key, value in msg_dict.items():
        if isinstance(value, float):
            msg_dict[key] = f'{value:.6f}'
        elif isinstance(value, paddle.Tensor):
            msg_dict[key] = f'{value.item():.6f}'
    msg = str(msg_dict)

    return msg


def train(cfg):
    """train"""
    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    paddle.seed(cfg.seed)

    # Load dataset
    transform = Compose([Normalize(mean=[127.5], std=[127.5], data_format='CHW')])
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_loader = paddle.io.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size, num_workers=cfg.num_workers, shuffle=False)
    loader = {'train': train_loader, 'val': val_loader}

    # Load model
    model = build_model(cfg.model)
    scheduler = build_scheduler(cfg.scheduler)
    optimizer = build_optimizer(cfg.optimizer, model, scheduler)
    loss_func = nn.CrossEntropyLoss(use_softmax=False)

    # Log
    logwriter = LogWriter(logdir=cfg.save_path)

    # Train
    step = 1
    timer = Timer(step, len(loader['train']) * cfg.num_epochs)
    for epoch in range(1, cfg.num_epochs + 1):
        # Train
        model.train()

        for images, labels in tqdm.tqdm(loader['train'], desc='Train: '):
            step_ = step * cfg.train_batch_size

            outputs = model(images)
            loss = loss_func(outputs, labels)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            timer.step()
            if step % cfg.print_freq == 0:
                lr = scheduler.get_lr()
                msg_dict = {
                    'mode': 'Train',
                    'epoch': epoch,
                    'step': step_,
                    'loss': loss,
                    'lr': lr,
                    'eta': timer.eta(),
                }
                msg = format_msg(msg_dict)
                logging.info(msg)
                print(msg)
                logwriter.add_scalar('loss', loss, step_)
                logwriter.add_scalar('lr', lr, step_)
            step += 1

        scheduler.step()

        # Save model
        if epoch % cfg.save_freq == 0:
            state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            model_file = os.path.join(cfg.save_path, f'epoch-{epoch:02d}.pdparams')
            latest_model_file = os.path.join(cfg.save_path, 'latest.pdparams')
            paddle.save(state_dict, model_file)
            shutil.copy(model_file, latest_model_file)

        # Val
        if cfg.evaluate or True:
            tp = 0
            model.eval()
            for images, labels in tqdm.tqdm(loader['val'], desc='Test: '):
                outputs = model(images)
                preds = paddle.argmax(outputs, axis=-1, keepdim=True)
                tp += (preds == labels).sum().item()
            accuracy = float(tp) / float(len(loader['val'].dataset))

            msg_dict = {
                'mode': 'Test',
                'epoch': epoch,
                'step': step,
                'accuracy': accuracy,
            }
            msg = format_msg(msg_dict)
            logging.info(msg)
            print(msg)
            logwriter.add_scalar('accuracy', accuracy, step_)

    logwriter.close()


if __name__ == '__main__':
    # Parameter
    cfg = config.parse()
    os.makedirs(cfg.save_path, exist_ok=True)

    # set base logging config
    fmt = '[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO, filename=os.path.join(cfg.save_path, 'log.txt'))
    logging.info(cfg)
    print(cfg)

    # Train network
    train(cfg)
