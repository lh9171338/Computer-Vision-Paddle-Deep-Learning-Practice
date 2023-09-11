# -*- encoding: utf-8 -*-
"""
@File    :   peach_dataset.py
@Time    :   2023/09/11 15:40:14
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms as T


class PeachDataset(Dataset):
    """PeachDataset"""

    def __init__(
        self,
        data_root,
        anno_file,
        mode,
        transform,
        **kwargs,
    ):
        super().__init__()

        self.data_root = data_root
        self.anno_file = anno_file
        self.mode = mode

        # 加载数据路径
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        self.image_file_list = [os.path.join(data_root, line.strip().split()[0]) for line in lines]
        self.label_list = [int(line.strip().split()[1]) for line in lines]
        
        # 数据增强
        if not isinstance(transform, dict):
            transform = dict(transform)
        if mode == 'train':
            self.transform = T.Compose(
                [
                    # T.ColorJitter(*transform['colorjitter']),
                    T.Resize(size=transform['image_size']),
                    T.RandomHorizontalFlip(prob=transform['horizontal_flip_prob']),
                    T.RandomVerticalFlip(prob=transform['vertical_flip_prob']),
                    T.ToTensor(),
                    T.Normalize(transform['mean'], transform['std']),
                ]
            )            
        else:
            self.transform = T.Compose(
                [
                    T.Resize(size=transform['image_size']),
                    T.ToTensor(),
                    T.Normalize(transform['mean'], transform['std']),
                ]
            )

    def __getitem__(self, index):
        # 加载图像
        image_file = self.image_file_list[index]
        image = cv2.imread(image_file).astype('float32')

        # 读取标签
        label = self.label_list[index]
        label = paddle.to_tensor([label]).astype('int64')

        # 数据增强
        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_file_list)


if __name__ == '__main__':
    dataset = {
        'type': 'PeachDataset',
        'data_root': 'data/peach-dataset',
        'anno_file': 'data/peach-dataset/train_list.txt',
        'mode': 'test',
        'transform': {
            'image_size': (224, 224),
            'colorjitter': [0.4, 0.4, 0.4, 0.4],
            'mean': [127.5, 127.5, 127.5],
            'std': [127.5, 127.5, 127.5],
            'horizontal_flip_prob': 0.5,
            'vertical_flip_prob': 0.5,
        },
    }
    dataset = PeachDataset(**dataset)
    image, label = dataset[0]
    print(image.shape, image.dtype, label)
