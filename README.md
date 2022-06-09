## Overview

In this branch, we provide another form of code, which obeys the design style of quantization structure in MQBench.

## Usage

Go into the exp/w2a4 directory. You can find config.yaml and run.sh for each arch. Execute the run.sh for quantized model. Other bit settings only need to change the corresponding bit number in yaml file.

## Results

Results on low-bit activation in terms of accuracy on ImageNet.
| Methods |  Bits (W/A) | Res18 | Res50 | MNV2 | Reg600M | Reg3.2G | MNasx2 |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   Full Prec. |   32/32 | 71.06 | 77.00 | 72.49 | 73.71 | 78.36 | 76.68   |
|QDrop| 4/4 | 69.13 | 75.12 | 67.83 | 70.95 | 76.46 | 73.04 |
|QDrop| 2/4 | 64.38 | 70.31 | 54.29 | 63.07 | 71.84| 63.28|
|QDrop| 3/3 | 65.68 | 71.28 | 54.38 | 64.65 | 71.69 | 64.05 |
|QDrop| 2/2 | 51.69 | 55.18 | 11.95 | 39.13 | 54.40 | 23.66 |


## Reference

If you find this repo useful for your research, please consider citing the paper:

    @article{wei2022qdrop,
	title={QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization},
	author={Wei, Xiuying and Gong, Ruihao and Li, Yuhang and Liu, Xianglong and Yu, Fengwei},
	journal={arXiv preprint arXiv:2203.05740},
	year={2022}
	}
