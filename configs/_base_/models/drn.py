'''
@Time    : 2022/5/19 15:47
@Author  : leeguandon@gmail.com
'''
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DRN',
        arch='d',
        depth=107,
        out_indices=(7,),
        deep_stem=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,)))
