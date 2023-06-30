# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class DosDataset(CustomDataset):
    CLASSES = ('_background_', 'faeces', 'socks', 'rope', 'plastic_bag')

    PALETTE = [[0, 0, 0], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64]]

    def __init__(self, **kwargs):
        super(DosDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
