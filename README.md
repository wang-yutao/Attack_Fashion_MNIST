# 图像分类模型的对抗攻击和对抗训练

题目详见https://github.com/LC-John/Fashion-MNIST

### 分类器

分类器源码详见 `my_train.py`，该模型使用 tensorflow 2.1.0 搭建，训练出的模型保存为 `my_model.h5`。该分类器在测试集上的准确率为 **90.85%**。

### 白盒攻击

在我的分类器上对 1000 个样本分别进行 1000 次白盒攻击，成功率为 **87.7%**。

10 组攻击成功的样本如下图所示，图片文件位于 `my_white_attack` 文件夹。

| <img src="my_white_attack/27_attack_2.jpg" alt="27_attack_2" style="zoom:200%;" /> | <img src="my_white_attack/46_attack_8.jpg" alt="46_attack_8" style="zoom:200%;" /> | <img src="my_white_attack/178_attack_3.jpg" alt="178_attack_3" style="zoom:200%;" /> | <img src="my_white_attack/204_attack_3.jpg" alt="204_attack_3" style="zoom:200%;" /> | <img src="my_white_attack/286_attack_3.jpg" alt="286_attack_3" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          2 Pullover                          |                            8 Bag                             |                           3 Dress                            |                           3 Dress                            |                           3 Dress                            |
| <img src="my_white_attack/27_correct_1.jpg" alt="27_correct_1" style="zoom:200%;" /> | <img src="my_white_attack/46_correct_7.jpg" alt="46_correct_7" style="zoom:200%;" /> | <img src="my_white_attack/178_correct_2.jpg" alt="178_correct_2" style="zoom:200%;" /> | <img src="my_white_attack/204_correct_2.jpg" alt="204_correct_2" style="zoom:200%;" /> | <img src="my_white_attack/286_correct_2.jpg" alt="286_correct_2" style="zoom:200%;" /> |
|                          1 Trouser                           |                          7 Sneaker                           |                          2 Pullover                          |                          2 Pullover                          |                          2 Pullover                          |

| <img src="my_white_attack/332_attack_3.jpg" alt="332_attack_3" style="zoom:200%;" /> | <img src="my_white_attack/463_attack_5.jpg" alt="463_attack_5" style="zoom:200%;" /> | <img src="my_white_attack/716_attack_5.jpg" alt="716_attack_5" style="zoom:200%;" /> | <img src="my_white_attack/810_attack_7.jpg" alt="810_attack_7" style="zoom:200%;" /> | <img src="my_white_attack/838_attack_6.jpg" alt="838_attack_6" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           3 Dress                            |                           5 Sandal                           |                           5 Sandal                           |                          7 Sneaker                           |                           6 Shirt                            |
| <img src="my_white_attack/332_correct_2.jpg" alt="332_correct_2" style="zoom:200%;" /> | <img src="my_white_attack/463_correct_4.jpg" alt="463_correct_4" style="zoom:200%;" /> | <img src="my_white_attack/716_correct_4.jpg" alt="716_correct_4" style="zoom:200%;" /> | <img src="my_white_attack/810_correct_6.jpg" alt="810_correct_6" style="zoom:200%;" /> | <img src="my_white_attack/838_correct_5.jpg" alt="838_correct_5" style="zoom:200%;" /> |
|                          2 Pullover                          |                            4 Coat                            |                            4 Coat                            |                           6 Shirt                            |                           5 Sandal                           |

### 黑盒攻击

在提供的分类器上对 1000 个样本分别进行 1000 次黑盒攻击，成功率为 **60.4%**。

10 组攻击成功的样本如下图所示，图片文件位于 `black_attack` 文件夹。

| <img src="black_attack/16_attack_3.jpg" alt="16_attack_3" style="zoom:200%;" /> | <img src="black_attack/96_attack_8.jpg" alt="96_attack_8" style="zoom:200%;" /> | <img src="black_attack/103_attack_9.jpg" alt="103_attack_9" style="zoom:200%;" /> | <img src="black_attack/560_attack_3.jpg" alt="560_attack_3" style="zoom:200%;" /> | <img src="black_attack/612_attack_6.jpg" alt="612_attack_6" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           3 Dress                            |                            8 Bag                             |                         9 Ankle boot                         |                           3 Dress                            |                           6 Shirt                            |
| <img src="black_attack/16_correct_2.jpg" alt="16_correct_2" style="zoom:200%;" /> | <img src="black_attack/96_correct_7.jpg" alt="96_correct_7" style="zoom:200%;" /> | <img src="black_attack/103_correct_8.jpg" alt="103_correct_8" style="zoom:200%;" /> | <img src="black_attack/560_correct_2.jpg" alt="560_correct_2" style="zoom:200%;" /> | <img src="black_attack/612_correct_5.jpg" alt="612_correct_5" style="zoom:200%;" /> |
|                          2 Pullover                          |                          7 Sneaker                           |                            8 Bag                             |                          2 Pullover                          |                           5 Sandal                           |

| <img src="black_attack/627_attack_3.jpg" alt="627_attack_3" style="zoom:200%;" /> | <img src="black_attack/661_attack_1.jpg" alt="661_attack_1" style="zoom:200%;" /> | <img src="black_attack/714_attack_9.jpg" alt="714_attack_9" style="zoom:200%;" /> | <img src="black_attack/774_attack_3.jpg" alt="774_attack_3" style="zoom:200%;" /> | <img src="black_attack/962_attack_6.jpg" alt="962_attack_6" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           3 Dress                            |                          1 Trouser                           |                         9 Ankle boot                         |                           3 Dress                            |                           6 Shirt                            |
| <img src="black_attack/627_correct_2.jpg" alt="627_correct_2" style="zoom:200%;" /> | <img src="black_attack/661_correct_0.jpg" alt="661_correct_0" style="zoom:200%;" /> | <img src="black_attack/714_correct_8.jpg" alt="714_correct_8" style="zoom:200%;" /> | <img src="black_attack/774_correct_2.jpg" alt="774_correct_2" style="zoom:200%;" /> | <img src="black_attack/962_correct_5.jpg" alt="962_correct_5" style="zoom:200%;" /> |
|                          2 Pullover                          |                        0 T-shirt/top                         |                            8 Bag                             |                          2 Pullover                          |                           5 Sandal                           |

在我的分类器上对 1000 个样本分别进行 1000 次黑盒攻击，成功率为 **37.5%**。

10 组攻击成功的样本如下图所示，图片文件位于 `my_black_attack` 文件夹。

| <img src="my_black_attack/18_attack_8.jpg" alt="18_attack_8" style="zoom:200%;" /> | <img src="my_black_attack/20_attack_1.jpg" alt="20_attack_1" style="zoom:200%;" /> | <img src="my_black_attack/273_attack_6.jpg" alt="273_attack_6" style="zoom:200%;" /> | <img src="my_black_attack/288_attack_0.jpg" alt="288_attack_0" style="zoom:200%;" /> | <img src="my_black_attack/321_attack_6.jpg" alt="321_attack_6" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                            8 Bag                             |                          1 Trouser                           |                           6 Shirt                            |                        0 T-shirt/top                         |                           6 Shirt                            |
| <img src="my_black_attack/18_correct_7.jpg" alt="18_correct_7" style="zoom:200%;" /> | <img src="my_black_attack/20_correct_0.jpg" alt="20_correct_0" style="zoom:200%;" /> | <img src="my_black_attack/273_correct_5.jpg" alt="273_correct_5" style="zoom:200%;" /> | <img src="my_black_attack/288_correct_9.jpg" alt="288_correct_9" style="zoom:200%;" /> | <img src="my_black_attack/321_correct_5.jpg" alt="321_correct_5" style="zoom:200%;" /> |
|                          7 Sneaker                           |                        0 T-shirt/top                         |                           5 Sandal                           |                         9 Ankle boot                         |                           5 Sandal                           |

| <img src="my_black_attack/326_attack_1.jpg" alt="326_attack_1" style="zoom:200%;" /> | <img src="my_black_attack/417_attack_0.jpg" alt="417_attack_0" style="zoom:200%;" /> | <img src="my_black_attack/610_attack_6.jpg" alt="610_attack_6" style="zoom:200%;" /> | <img src="my_black_attack/647_attack_1.jpg" alt="647_attack_1" style="zoom:200%;" /> | <img src="my_black_attack/981_attack_1.jpg" alt="981_attack_1" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          1 Trouser                           |                        0 T-shirt/top                         |                           6 Shirt                            |                          1 Trouser                           |                          1 Trouser                           |
| <img src="my_black_attack/326_correct_0.jpg" alt="326_correct_0" style="zoom:200%;" /> | <img src="my_black_attack/417_correct_9.jpg" alt="417_correct_9" style="zoom:200%;" /> | <img src="my_black_attack/610_correct_5.jpg" alt="610_correct_5" style="zoom:200%;" /> | <img src="my_black_attack/647_correct_0.jpg" alt="647_correct_0" style="zoom:200%;" /> | <img src="my_black_attack/981_correct_0.jpg" alt="981_correct_0" style="zoom:200%;" /> |
|                        0 T-shirt/top                         |                         9 Ankle boot                         |                           5 Sandal                           |                        0 T-shirt/top                         |                        0 T-shirt/top                         |

### 对抗训练

在新分类器上进行白盒攻击，成功率为 **75.7%**。其中 10 组攻击成功的样本如下图所示，图片文件位于 `my_new_white_attack` 文件夹。

| <img src="my_new_white_attack/155_attack_1.jpg" alt="155_attack_1" style="zoom:200%;" /> | <img src="my_new_white_attack/429_attack_4.jpg" alt="429_attack_4" style="zoom:200%;" /> | <img src="my_new_white_attack/695_attack_4.jpg" alt="695_attack_4" style="zoom:200%;" /> | <img src="my_new_white_attack/779_attack_4.jpg" alt="779_attack_4" style="zoom:200%;" /> | <img src="my_new_white_attack/784_attack_3.jpg" alt="784_attack_3" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          1 Trouser                           |                            4 Coat                            |                            4 Coat                            |                            4 Coat                            |                           3 Dress                            |
| <img src="my_new_white_attack/155_correct_0.jpg" alt="155_correct_0" style="zoom:200%;" /> | <img src="my_new_white_attack/429_correct_3.jpg" alt="429_correct_3" style="zoom:200%;" /> | <img src="my_new_white_attack/695_correct_3.jpg" alt="695_correct_3" style="zoom:200%;" /> | <img src="my_new_white_attack/779_correct_3.jpg" alt="779_correct_3" style="zoom:200%;" /> | <img src="my_new_white_attack/784_correct_2.jpg" alt="784_correct_2" style="zoom:200%;" /> |
|                        0 T-shirt/top                         |                           3 Dress                            |                           3 Dress                            |                           3 Dress                            |                          2 Pullover                          |

| <img src="my_new_white_attack/892_attack_8.jpg" alt="892_attack_8" style="zoom:200%;" /> | <img src="my_new_white_attack/899_attack_7.jpg" alt="899_attack_7" style="zoom:200%;" /> | <img src="my_new_white_attack/907_attack_6.jpg" alt="907_attack_6" style="zoom:200%;" /> | <img src="my_new_white_attack/942_attack_3.jpg" alt="942_attack_3" style="zoom:200%;" /> | <img src="my_new_white_attack/957_attack_4.jpg" alt="957_attack_4" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                            8 Bag                             |                          7 Sneaker                           |                           6 Shirt                            |                           3 Dress                            |                            4 Coat                            |
| <img src="my_new_white_attack/892_correct_7.jpg" alt="892_correct_7" style="zoom:200%;" /> | <img src="my_new_white_attack/899_correct_6.jpg" alt="899_correct_6" style="zoom:200%;" /> | <img src="my_new_white_attack/907_correct_5.jpg" alt="907_correct_5" style="zoom:200%;" /> | <img src="my_new_white_attack/942_correct_2.jpg" alt="942_correct_2" style="zoom:200%;" /> | <img src="my_new_white_attack/957_correct_3.jpg" alt="957_correct_3" style="zoom:200%;" /> |
|                          7 Sneaker                           |                           6 Shirt                            |                           5 Sandal                           |                          2 Pullover                          |                           3 Dress                            |

在新分类器上进行黑盒攻击，成功率为 **88.6%**。其中 10 组攻击成功的样本如下图所示，图片文件位于 `my_new_black_attack` 文件夹。

| <img src="my_new_black_attack/98_attack_8.jpg" alt="98_attack_8" style="zoom:200%;" /> | <img src="my_new_black_attack/153_attack_2.jpg" alt="153_attack_2" style="zoom:200%;" /> | <img src="my_new_black_attack/224_attack_0.jpg" alt="224_attack_0" style="zoom:200%;" /> | <img src="my_new_black_attack/260_attack_6.jpg" alt="260_attack_6" style="zoom:200%;" /> | <img src="my_new_black_attack/269_attack_4.jpg" alt="269_attack_4" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                            8 Bag                             |                          2 Pullover                          |                        0 T-shirt/top                         |                           6 Shirt                            |                           5 Sandal                           |
| <img src="my_new_black_attack/98_correct_7.jpg" alt="98_correct_7" style="zoom:200%;" /> | <img src="my_new_black_attack/153_correct_1.jpg" alt="153_correct_1" style="zoom:200%;" /> | <img src="my_new_black_attack/224_correct_9.jpg" alt="224_correct_9" style="zoom:200%;" /> | <img src="my_new_black_attack/260_correct_5.jpg" alt="260_correct_5" style="zoom:200%;" /> | <img src="my_new_black_attack/269_correct_3.jpg" alt="269_correct_3" style="zoom:200%;" /> |
|                          7 Sneaker                           |                          1 Trouser                           |                         9 Ankle boot                         |                           5 Sandal                           |                            4 Coat                            |

| <img src="my_new_black_attack/456_attack_5.jpg" alt="456_attack_5" style="zoom:200%;" /> | <img src="my_new_black_attack/552_attack_8.jpg" alt="552_attack_8" style="zoom:200%;" /> | <img src="my_new_black_attack/581_attack_2.jpg" alt="581_attack_2" style="zoom:200%;" /> | <img src="my_new_black_attack/617_attack_0.jpg" alt="617_attack_0" style="zoom:200%;" /> | <img src="my_new_black_attack/849_attack_5.jpg" alt="849_attack_5" style="zoom:200%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           5 Sandal                           |                            8 Bag                             |                          2 Pullover                          |                        0 T-shirt/top                         |                           5 Sandal                           |
| <img src="my_new_black_attack/456_correct_4.jpg" alt="456_correct_4" style="zoom:200%;" /> | <img src="my_new_black_attack/552_correct_7.jpg" alt="552_correct_7" style="zoom:200%;" /> | <img src="my_new_black_attack/581_correct_1.jpg" alt="581_correct_1" style="zoom:200%;" /> | <img src="my_new_black_attack/617_correct_9.jpg" alt="617_correct_9" style="zoom:200%;" /> | <img src="my_new_black_attack/849_correct_4.jpg" alt="849_correct_4" style="zoom:200%;" /> |
|                            4 Coat                            |                          7 Sneaker                           |                          1 Trouser                           |                         9 Ankle boot                         |                            4 Coat                            |

### 总结

新旧分类器的测试集准确率以及白盒黑盒攻击成功率如下表所示。

|                | 旧分类器 | 新分类器 |
| :------------: | :------: | :------: |
|  测试集准确率  |  90.85%  |  89.13%  |
| 白盒攻击成功率 |  87.7%   |  75.7%   |
| 黑盒攻击成功率 |  37.5%   |  88.6%   |

- 黑盒攻击所得样本噪声较大，而白盒攻击所得样本与原图相差无异。
- 通过对抗训练，新分类器在测试集上的准确率降低了，说明训练集的训练难度增加了。
- 在新分类器上，白盒攻击的成功率下降了，说明新分类器的性能和鲁棒性有所提升。
- 但对于新分类器的黑盒攻击成功率反而提高了，这也许是提高白盒攻击鲁棒性的负面影响，也可能是模型设置或攻击模式的外在原因。
