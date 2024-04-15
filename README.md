# BRCD

A Pytorch implementation of paper Bit-mask Robust Contrastive Knowledge Distillation for Unsupervised Semantic Hashing (WWW2024) 

# Main Dependencies

+ pytorch               1.10.1
+ torchvision         0.11.2
+ numpy               1.19.5
+ pandas              1.1.5
+ Pillow              8.4.0

# How to run

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
