# QDrop

## Introduction

This branch supports our quantization algorithm *QDrop* on detection task with MS COCO dataset.

## File Organization

```
QDrop
├── mmdet/       [mmdetection tools]
├── configs/     [mmdetection configs including datasets, models, and schedules]
├── qdrop/
|   ├── solver         
|   |   ├──quant_coco.py [Entrance of the code]
|   |   ├──recon.py [reconstruct one layer or one block]
|   ├── model/     [quantized models]
|   ├── quantization/ [quantization tools]
```
## Installation

The task part is based on [mmdetection](https://github.com/open-mmlab/mmdetection). We use version of v2.28.2. Please refer to its [readme](README_mmdetection.md) for details.


    # Requirements
    pytorch=1.11.0
    mmcv-full==1.4.7

FP models are downloaded from its [model zoos](https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn). We choose Faster-RCNN and RetinaNet, and take *R-50-FPN pytorch 2x* and *R-101-FPN pytorch 2x* as pretrained weights.


## Usage

Go into the exp directory. You can see the *slurm_test.sh* and *config.yaml* files here. *config.yaml* indicates quantization settings. Before running the code, you need to change the parition, task config path and FP model path in *slurm_test.sh*.

```
# slurm_test.sh
set -x
q_config=$1 # quantization configs
work_dir=$2 # output dir
PARTITION=toolchain
JOB_NAME=coco

CONFIG=configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py
# configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py
# path to task configs
CHECKPOINT=/mnt/lustre/weixiuying/model_zoo/mscoco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
#/mnt/lustre/weixiuying/model_zoo/mscoco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
# path to FP model path
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u qdrop/solver/quant_coco.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" --work-dir=${work_dir} --fuse-conv-bn --eval='bbox' --q_config=${q_config}

```

```
a_qconfig:
    quantizer: LSQFakeQuantize
    observer: AvgMSEFastObserver
    bit: 4 # bit selection
    symmetric: False
    ch_axis: -1
w_qconfig:
    quantizer: AdaRoundFakeQuantize
    observer: MSEObserver
    bit: 2 # bit selection
    symmetric: False
    ch_axis: 0
calibrate: 256
recon:
    batch_size: 2
    scale_lr: 4.0e-5
    warm_up: 0.2
    weight: 0.01
    iters: 20000
    b_range: [20, 2]
    keep_gpu: True
    round_mode: learned_hard_sigmoid
    drop_prob: 0.5
```
Now, you can run the code as follows:
```
./exp/slurm_test.sh exp/config.yaml exp/results/w2a4/faster_rcnn_r50
```
## Results

Results on low-bit activation in terms of box AP on MS COCO.

| Methods    | Bits (W/A) | Faster R-CNN | Faster R-CNN | RetinaNet  | RetinaNet   |
| ---------- | ---------- | ------------ | ------------ | ---------- | ----------- |
|            |            | R50-FPN-2x   | R101-FPN-2x  | R50-FPN-2x | R101-FPN-2x |
| Full Prec. | 32/32      | 38.4         | 39.8         | 37.4       | 38.9        |
| QDrop      | 4/4        | 37.0         | 38.0         | 35.8       | 37.3        |
| QDrop      | 2/4        | 35.0         | 36.2         | 34.0       | 35.8        |


## Reference

If you find this repo useful for your research, please consider citing the paper:

    @article{wei2022qdrop,
    title={QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization},
    author={Wei, Xiuying and Gong, Ruihao and Li, Yuhang and Liu, Xianglong and Yu, Fengwei},
    journal={arXiv preprint arXiv:2203.05740},
    year={2022}
    }