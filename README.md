# Two-CILs

This is the official repository for the CVPR 2021 Continual Learning workshop paper: "_A Tale of Two CILs: The
Connections between Class Incremental Learning and Class Imbalanced Learning, and Beyond_"
[[Paper]](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/papers/He_A_Tale_of_Two_CILs_The_Connections_Between_Class_Incremental_CVPRW_2021_paper.pdf)
[[Supp]](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/supplemental/He_A_Tale_of_CVPRW_2021_supplemental.pdf)

You can perform the following three steps to run our codes!

<!---
## Abstract

Catastrophic forgetting, the main challenge of Class Incremental Learning, is closely related to the classifier’s bias
due to imbalanced data, and most researchers resort to empirical techniques to remove the bias. Such anti-bias tricks
share many ideas with the field of Class Imbalanced Learning, which encourages us to reflect on why these tricks work,
and how we can design more principled solutions from a different perspective. In this paper, we comprehensively analyze
the connections and seek possible collaborations between these two fields, i.e. Class Incremental Learning and Class
Imbalanced Learning. Specifically, we first provide a panoramic view of recent bias correction tricks from the
perspective of handling class imbalance. Then, we show that an adapted post-scaling technique which originates from
Class Imbalanced Learning is on par with or even outperforms SOTA Class Incremental Learning method. Visualization via
violin plots and polar charts further sheds light on how SOTA methods address the class imbalance problem from a more
intuitive geometric perspective. These findings may encourage further infiltration between the two closely connected
fields, but also raise concerns about whether it is correct that Class Incremental Learning degenerates into a class
imbalance problem.
-->

## 1. Requirements

The code is implemented in Python 3.6.

As for CUDA, we use CUDA 10.1 and cuDNN 7.6.

For requirements for the Python modules, you can simply run:

``pip install -r requirements.txt``

## 2. Datasets

For **CIFAR-100**, download the [python version of CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
and extract it in a certain folder, let's say `/home/user/cifar-100-python`, then set `data_path` of _cifar100.py_ to
this folder.

For **Group ImageNet**, download the
ImageNet64x64 ([[Google Drive]](https://drive.google.com/drive/folders/1ESWOB1C7pHjNOH12Uo0LQXPGnv7MbvPo?usp=sharing) [[百度网盘 (提取码:nqi7)]](https://pan.baidu.com/s/1KlXN-id7ybXn-zYZJv06BA)
dataset first and change `data_path` in _imagenet64x64.py_ to your folder path. ImageNet64x64 is a downsampled ImageNet
according to https://patrykchrabaszcz.github.io/Imagenet32/.

## 3. Usage

After downloading the datasets and changing the `data_path`, simply run the following scripts for **CIFAR-100** and
**Group ImageNet** respectively (some configurations in the script should be set in advance, e.g. ``LD_LIBRARY_PATH``):

`bash scripts/cifar100.sh`

`bash scripts/imagenet64x64.sh`

By default, the scripts above use LeNet on CIFAR-100 and ResNet-18 on Group ImageNet. You can change
the ``COMMON_FLAGS`` if you want to try other settings.

After running all experiments above, you can run the following scripts to show the top-1 accuracy and running time of
all methods:

`bash scripts/show_cifar100.sh`

`bash scripts/show_imagenet64x64.sh`

To obtain the violin graphs and the polar charts, simply go to the result folders of _post-scaling_ and _MDFCIL_, which
are
in ``result/cifar100_random_1/base_10_inc_10_total_100/seed_1993/skip_first_resnet_70_adam_0.005_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_post_scaling``
and ``result/imagenet64x64_10x10_random_1/base_10_inc_10_total_100/seed_1993/skip_first_resnet18_70_adam_0.005_aug_wd_0.0001_random_20_total_lwf_1.0_temp_2.0_adj_w_weight_aligning_no_bias``
respectively.

[//]: <> (To obtain the violin graphs and the polar charts, p)

## Citation

If you use these codes, please cite our paper:

```bibtex
@inproceedings{he2021tale,
  title={A Tale of Two CILs: The Connections Between Class Incremental Learning and Class Imbalanced Learning, and Beyond},
  author={He, Chen and Wang, Ruiping and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={3559--3569},
  year={2021}
}
```

## Note

You can regard this repository as a Tensorflow library for Class Incremental Learning methods, a counterpart
for [Avalanche](https://avalanche.continualai.org/) or [Sequoia](https://github.com/lebrice/Sequoia) implemented by
PyTorch. A more comprehensive version can be found in [TF2-CIL](https://github.com/TonyPod/TF2-CIL).

However, the main purpose of this work is to compare the anti-biasing techniques in recent Class Incremental Learning
methods. Thus, unfair techniques such as cutmix, intense data augmentation are removed.

If your goal is to fully reproduce results listed in other papers, then this repository might not be the optimal
solution.

## Further

If you have any question, feel free to contact me. My email is _chen.he@vipl.ict.ac.cn_