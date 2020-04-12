# Human Strategic Behavior Prediction

## Introduction

Standard game-theoretic frameworks like Nash Equilibrium assume that all the agents are perfectly rational in decision making, which results in inferior prediction performance for real-world data. In this repository we provide a deep learning toolbox for predicting the human stategic behavior in normal form games (i.e., the distribution of players' decisions). This package is built with the PyTorch framework.

## Training

```
CUDA_VISIBLE_DEVICES=0 python -u main.py --num-kernels 64
```

## Inference

```
CUDA_VISIBLE_DEVICES=0 python -u main.py --num-kernels 64 --inference
```