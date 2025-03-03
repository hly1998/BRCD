# BRCD

A Pytorch implementation of paper Bit-mask Robust Contrastive Knowledge Distillation for Unsupervised Semantic Hashing (WWW2024) (https://arxiv.org/pdf/2403.06071).

### Overview

This paper proposes **Bit-aware Robust Contrastive Knowledge Distillation (BRCD)**, a method specifically designed for the distillation of unsupervised semantic hashing models, aiming to mitigate the inference latency caused by large-scale backbone networks such as **ViT**. Our approach first aligns the semantic spaces of the teacher and student models through a contrastive learning objective, achieving knowledge distillation at both the individual feature level and structural semantic level, thereby ensuring the effectiveness of two key search paradigms in semantic hashing. Furthermore, we incorporate a clustering-based strategy within the contrastive learning objective to eliminate noise augmentation and ensure robust optimization. Additionally, through bit-level analysis, we reveal the redundancy issue in hash codes caused by bit independence and introduce a **bit masking mechanism** to alleviate its impact. Extensive experiments demonstrate that **BRCD** exhibits superior performance and strong generalizability across various semantic hashing models and backbone networks, significantly outperforming existing knowledge distillation methods.

![framwork](image/framework_7.png)

### Main Dependencies

+ pytorch             1.10.1
+ torchvision         0.11.2
+ numpy               1.19.5
+ pandas              1.1.5
+ Pillow              8.4.0

### How to run

We show how to run our code in CIFAR-10 dataset when using ViT_b_16 as the teacher model.

First, run following command to train a teacher model:

```
sh scripts/run.sh
```

If you run the above command, the program will download the CIFAR-10 dataset to the directory ./data/cifar10/ and then start to train.

Then, train the student model using our BRCD method:

```
sh scripts/ours_distill_run.sh
```

### Citation
If you find our code useful and use BRCD in your work, please cite our paper.
```bash
@inproceedings{he2024bit,
  title={Bit-mask Robust Contrastive Knowledge Distillation for Unsupervised Semantic Hashing},
  author={He, Liyang and Huang, Zhenya and Liu, Jiayu and Chen, Enhong and Wang, Fei and Sha, Jing and Wang, Shijin},
  booktitle={Proceedings of the ACM Web Conference 2024},
  pages={1395--1406},
  year={2024}
}
```