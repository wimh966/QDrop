#!/usr/bin/env bash

set -x
q_config=$1
work_dir=$2
PARTITION=toolchain
JOB_NAME=coco

CONFIG=configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py
# configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py
CHECKPOINT=/mnt/lustre/weixiuying/model_zoo/mscoco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth
#'/mnt/lustre/weixiuying/model_zoo/mscoco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
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
