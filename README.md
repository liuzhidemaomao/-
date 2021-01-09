# -
## 课程作业

代码包含了课程报告中的一阶段和二阶段的过程。其中，第二阶段在运行时单张卡大概需要15-16G，一阶段大概需要9G-10G。

## 训练数据集是cityscapes

官方下载:
https://www.cityscapes-dataset.com/

## 预训练的resnet和其他

其中，pretrained_models文件夹中包含训练好的模型权重；one_stage和two_stage文件夹中分别包含训练好的模型权重

百度网盘：
链接：https://pan.baidu.com/s/1pzANm8Cb60T6MPVKTNCfmw 
提取码：lqht 

## 文件夹结构

````bash
$HRNet-Semantic-Segmentation-pytorch1.1
├── data
│   ├── cityscapes
│   │   ├── gtFine
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   └── leftImg8bit
│   │       ├── test
│   │       ├── train
│   │       └── val
│   ├── list
│   │   ├── cityscapes
│   │   │   ├── test.lst
│   │   │   ├── trainval.lst
│   │   │   └── val.lst
├── pretrained_models
│   ├── resnet50_v2.pth
│   ├── resnet152_v2.pth
├── lib
├── experiments
├── tools
````

## 环境

numpy==1.19.2
torch==1.1.0
yacs==0.1.8
tqdm==4.51.0
opencv_python==3.4.1.15
scikit_image==0.17.2
Pillow==8.1.0
skimage==0.0
tensorboardX==2.1

## 训练和测试
训练代码，其中nproc_per_node和使用的卡的数量一致
````bash
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

测试代码

在test时，需要将配置文件seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml修改成如下形式(0,)， 或者使用两张卡来跑
````bash
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

## Reference
本代码主要是基于HRNet和PSPnet的工作

[1] Wang J, Sun K, Cheng T, et al. Deep high-resolution representation learning for visual recognition[J]. IEEE transactions on pattern analysis and machine intelligence, 2020.  [download](https://ieeexplore.ieee.org/abstract/document/9052469/)

[2] Zhao H, Shi J, Qi X, et al. Pyramid scene parsing network[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2881-2890.  [download](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)
