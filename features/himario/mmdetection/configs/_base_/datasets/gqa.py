dataset_type = 'GQA'
data_root = '/Disk2/GQA/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_attrs=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AttributeFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_attrs']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_A_train = dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/gqa_objects.train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline)
)
dataset_A_val = dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/gqa_objects.val.json',
    img_prefix=data_root + 'images/',
    pipeline=test_pipeline
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dataset_A_train,
    val=dataset_A_val,
    test=dataset_A_val)

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/gqa_objects.train.json',
#         img_prefix=data_root + 'images/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/gqa_objects.val.json',
#         img_prefix=data_root + 'images/',
#         pipeline=test_pipeline),
#     test=None)

evaluation = dict(interval=1, metric='mAP')
