import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import register_datasets
from configs.mrcnn_config import get_config
from runner import get_trainer, get_dtrn_cfg

import random


def train(cfg):
    register_datasets(cfg)
    dtrn_cfg = get_dtrn_cfg(cfg)
    trainer = get_trainer(dtrn_cfg)
    trainer.train()


if __name__ == '__main__':
    cfg = get_config()
    random.seed(cfg.seed)
    train(cfg)