# QDrop
PyTorch implementation of QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization

## Overview

QDrop is a simple yet effective approach, which randomly drops the quantization of activations during reconstruction to pursue flatter model on both calibration and test data. QDrop is easy to implement for various neural networks including CNNs and Transformers, and plug-and-play with little additional computational complexity.

Experiments on various tasks include computer vision (image classification, object detection) and natural language processing (text classification and question answering) . 

## Integrated into MQBench
Our method has been integrated into quantization benchmark [MQBench](https://github.com/ModelTC/MQBench). And the docs can be found here <http://mqbench.tech/assets/docs/html/>

## Usage

Go into the exp directory and you can see run.sh and config.sh. run.sh represents a example for resnet18 w2a2. You can run config.sh to produce similar scripts across bits and archs.

run.sh
```
#!/bin/bash
PYTHONPATH=../../../../:$PYTHONPATH \
python ../../../main_imagenet.py --data_path data_path \
--arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2  --act_quant --order together --wwq --waq --awq --aaq \
--weight 0.01 --input_prob 0.5 --prob 0.5
```

config.sh

```
#!/bin/bash
# pretrain models and hyperparameters following BRECQ
arch=('resnet18' 'resnet50' 'mobilenetv2' 'regnetx_600m' 'regnetx_3200m' 'mnasnet')
weight=(0.01 0.01 0.1 0.01 0.01 0.2)
w_bit=(3 2 2 4)
a_bit=(3 4 2 4)
for((i=0;i<6;i++))
do
	for((j=0;j<4;j++))
	do
		path=w${w_bit[j]}a${a_bit[j]}/${arch[i]}
		mkdir -p $path
		echo $path
		cp run.sh $path/run.sh
		sed -re "s/weight([[:space:]]+)0.01/weight ${weight[i]}/" -i $path/run.sh
		sed -re "s/resnet18/${arch[i]}/" -i $path/run.sh
		sed -re "s/n_bits_w([[:space:]]+)2/n_bits_w ${w_bit[j]}/" -i $path/run.sh
		sed -re "s/n_bits_a([[:space:]]+)2/n_bits_a ${a_bit[j]}/" -i $path/run.sh
	done
done
```
Then you can get a series of scripts and run it directly to get the following results.
## Results

Results on low-bit activation in terms of accuracy on ImageNet. * represents for our implementation according to open-source codes.

| Methods |  Bits (W/A)    | ResNet-18     |ResNet-50| MobileNetV2 | RegNet-600MF | RegNet-3.2GF | MNasNet-2.0 |
| ------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   Full Prec.      |   32/32 | 71.06 | 77.00 | 72.49 | 73.71 | 78.36 | 76.68   |
|QDrop| 4/4 | 69.10 | 75.03 | 67.89 | 70.62 | 76.33 | 72.39 |
|QDrop| 2/4 | 64.66 | 70.08 | 52.92 | 63.10 | 70.95 | 62.36|
|QDrop| 3/3 | 65.56 | 71.07 | 54.27 | 64.53 | 71.43 | 63.47|
|QDrop| 2/2 | 51.14| 54.74| 8.46| 38.90| 52.36| 22.70|


## More experiments

**Case 1, Case 2, Case 3**

To compare the results of 3 Cases mentioned in the observation part of the method, we can use the following commands.

Case 1: weight tuning  doesn't feel any activation quantization

Case 2: weight tuning feels full activation quantization

Case 3: weight tuning feels part activation quantization

```
# Case 1
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order after --wwq --awq --aaq --input_prob 1.0 --prob 1.0
# Case 2
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order before --wwq --waq --aaq --input_prob 1.0 --prob 1.0
# Case 3
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order after --wwq --waq --awq --aaq --input_prob 1.0 --prob 1.0
```

**No Drop**

To compare with QDrop, No Drop can be achieved by turning the probability to 1.0 to disable dropping quantization during weight tuning.

```
python main_imagenet.py --data_path data_path --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 2 --act_quant --order together --wwq --waq --awq --aaq --input_prob 1.0 --prob 1.0
```
