# 腾讯觅影比赛小结

## 模型简述

![](https://raw.githubusercontent.com/eshoyuan/pic/main/20211115114921.png)



## 做过的尝试

### 有用

表现比较好的backbone: swin transformer base, seresnext系列, efficientnetv2

TTA: 软硬投票效果差不多

较大的分辨率

模型集成

标签平滑: 可以解决PPT中提到的验证集损失上升的问题, 但是对于准确率帮助不大, 关于这个问题可以看https://www.zhihu.com/question/318399418/answer/1202932315

model.eval(): 实验下来推理的时候应该是比model.train()好一点

### 没用

表现比较差的backbone: effcientent(b0-b7都有试, 效果都不行, 可能需要一些训练技巧)

学习率warm up

优化器调参: 这个确实没有仔细尝试, 但是在有限的尝试内, Adam, SGD以及学习率衰减, 只要不是初始学习率设置的太大, 都差不多.

triplet loss: 准确率基本不变

xgboost: 准确率基本不变

引入外部数据: Kaggle DR+单标签的数据扩充数据集后表现略有下降, 引入了类别不平衡问题, 而且Kaggle DR+的数据集标注似乎也不是特别准确. 考虑到引入后训练时间成倍增加, 并没有深入研究了, 这么多数据如果合理运用肯定会有所帮助. 

softmax损失改为bce: 准确率基本不变

不用imagenet的预训练参数: 准确率基本不变, 收敛速度似乎也差不多

修改池化层: 准确率基本不变

大batch size: 早期实验准确率是会有一点下降的

## 有可能有用, 但没试

mixup, cutmix, Auto Augment等增强手段.

triplet loss和svm/xgboost结合



## 相关资料

以下是认为对我帮助比较大的资料



本科生晋升GM记录 & kaggle比赛进阶技巧分享 https://zhuanlan.zhihu.com/p/93806755

这个资料在比赛前期为我提供了比较大的帮助



Kaggle APTOS2019 https://www.kaggle.com/c/aptos2019-blindness-detection/discussion?sort=votes 

Kaggle Cassava Leaf Disease Classification(9个月前结束) https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion?sort=votes

以上比赛的讨论区都有非常多的分享, 从backbone的选取到一些小技巧, APTOS与本次比赛内容相关度更大, 本次增强方法就是从这里借鉴的,  Cassava Leaf Disease Classification则更新, swin transformer的尝试也是看了其中一个金牌的解决方案后开始尝试.



Kaggle APTOS2019 11th solution https://github.com/4uiiurz1/kaggle-aptos2019-blindness-detection 这份代码比较简单易懂, 写的比我规范很多, 我也借鉴了他的一些内容和写法. 



timm模型库 https://rwightman.github.io/pytorch-image-models/

这个库非常好用, 上手容易, 模型齐全. 此外提一下这个库作者最近的一篇工作https://arxiv.org/abs/2110.00476, 利用各种先进的trick将ResNet50从75.3提升至80.4, 里面提到的方法对于此类比赛应该会有一些帮助.



CAM https://github.com/jacobgil/pytorch-grad-cam

根据示例就可以很快上手



集成和蒸馏 https://www.microsoft.com/en-us/research/blog/three-mysteries-in-deep-learning-ensemble-knowledge-distillation-and-self-distillation/

这篇博文是作者关于集成, 蒸馏, 自蒸馏的一些实验和看法, 让我对于集成和蒸馏有了新的认识.





