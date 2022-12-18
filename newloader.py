import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmcv.runner import HOOKS, build_optimizer, build_runner, get_dist_info
from mmcv.utils import build_from_cfg

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import find_latest_checkpoint, get_root_logger

data_loaders = [
    build_dataloader(
        ds,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True) for ds in dataset
]