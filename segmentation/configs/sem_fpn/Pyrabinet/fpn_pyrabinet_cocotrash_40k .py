_base_ = [
    '../../_base_/models/fpn_r50.py',
    '../../_base_/datasets/coco_trash.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='pyrabinet',
        style='pytorch'),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=5))


gpu_multiples = 1
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=4000//gpu_multiples)
evaluation = dict(interval=4000//gpu_multiples, metric='mIoU')
