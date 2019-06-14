# PSMNet-Tensorflow
=======
>>>>>>> PMSNet-Tensorflow

# Pyramid Stereo Matching Network

根据 "[Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669)" paper (CVPR 2018) 在[源代码](https://github.com/JiaRenChang/PSMNet)基础上使用tensorflow进行移植（源代码使用的pytorch）

### Citation
```
@inproceedings{chang2018pyramid,
  title={Pyramid Stereo Matching Network},
  author={Chang, Jia-Ren and Chen, Yong-Sheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5410--5418},
  year={2018}
}
```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results](#results)
4. [Contacts](#contacts)

## Introduction

Recent work has shown that depth estimation from a stereo pair of images can be formulated as a supervised learning task to be resolved with convolutional neural networks (CNNs). However, current architectures rely on patch-based Siamese networks, lacking the means to exploit context information for finding correspondence in illposed regions. To tackle this problem, we propose PSMNet, a pyramid stereo matching network consisting of two main modules: spatial pyramid pooling and 3D CNN. The spatial pyramid pooling module takes advantage of the capacity of global context information by aggregating context in different scales and locations to form a cost volume. The 3D CNN learns to regularize cost volume using stacked multiple hourglass networks in conjunction with intermediate supervision.

<img align="center" src="https://user-images.githubusercontent.com/11732099/43501836-1d32897c-958a-11e8-8083-ad41ec26be17.jpg">

## Usage

### Dependencies

- Python3.6
- Tensorflow(1.3.0)
- PIL
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

## 目前已经完成的工作
- 移植了KITTI2012数据集的读取工作
- 移植了preprocess中的部分函数
- 移植了KITTILoader，满足基本的数据读取
- 在tensorflow框架下完成了CNN子模块
- 在tensorflow框架下完成了SPP子模块
- 在tensorflow框架下完成了CNN3D子模块（这里只重写了论文中提到的stacked hourglass结构）
- 在tensorflow框架下完成了视差回归和损失函数
- 对main函数进行改写，满足输入输出需求

## 接下来的的工作
- 完善整体的model
- 加入论文中basic的模型
- 加入模型的保存和读取模块
- 加入tensorboard可视化需要的操作
- 完善输入和输出
- 整体进行训练

## ---------------------------------
## 2019-06-12更新
- 完善整体的model
- 加入训练SceneFlow数据集与KITTI数据集模块（data_loader为读取KITTI数据集，load_SceneFlow为读取SceneFlow数据集，由于SceneFlow数据集中，视差图为PFM格式，需要注意，为了加快数据读取的速度，在训练SceneFlow数据集时，首先需要运行generate_image_list.py得到整个数据集的地址）
- 完成模型的保存和读取模块
- 完成tensorboard可视化需要的操作
- 完成输入和输出
- 可以进行整体进行训练
