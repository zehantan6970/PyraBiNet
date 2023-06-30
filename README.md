# PyraBiNet

PyraBiNet is an innovative hybrid model optimized for lightweight semantic segmentation tasks. This model ingeniously merges the merits of Convolutional Neural Networks (CNNs) and Transformers.

## Data Preparation
1. Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision datasets. ImageFolder, and the training and validation data are expected to be in the train/ folder and val folder respectively:
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
2. Download the DOS dataset, organized in the following format:
├── data
│   ├── coco_trash
│   │   ├── images
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── annotations
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val

## Download

The DOS dataset can be downloaded from:  [DOS DATASET](https://github.com/zehantan6970/DOS_Dataset ) 

## Installation

Our project has been tested on torch=1.7.0 torchvision=0.8.1 timm=0.3.2

## Pre-Training

Train 300epochs on the ImageNet dataset with 8 GPUs:
sh dist_train.sh configs/sem_fpn/Pyrabinet/fpn_pyrabinet_ade20k_80k.py 8 --data-path /path/to/imagenet

## Training

Use 4 gpus to train pyrabinet+Semantic FPN on the ADE20K dataset, run:
dist_train.sh configs/sem_fpn/Pyrabinet/fpn_pyrabinet_ade20k_80k.py 4

## Evaluation

dist_test.sh configs/sem_fpn/Pyrabinet/fpn_pyrabinet_ade20k_80k.py /path/to/checkpoint_file 4 --out results.pkl --eval mIoU

## Calculating FLOPS & Params
python flops.py
