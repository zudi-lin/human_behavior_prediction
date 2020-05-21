# Human Strategic Behavior Prediction

## Introduction

Standard game-theoretic frameworks like Nash Equilibrium assume that all the agents are perfectly rational in decision making, which results in inferior prediction performance for real-world data. In this repository we provide a deep learning toolbox for predicting the human stategic behavior in the normal-form games (i.e., the distribution of players' decisions). This package is built with the PyTorch framework. For more information please see
the [blog post](https://medium.com/analytics-vidhya/predicting-human-strategies-in-games-via-deep-learning-787ae6667aca).

## Cross-validation

For all labeled samples, we conduct N-fold (N=5 by default) cross-validation experiments to decide the best model:
```
CUDA_VISIBLE_DEVICES=0 scripts/main.py --config-file configs/Bi-Matrix-V2.yaml \
--output results --mode cross_validate
```

## Training & Inference

After cross-validation, we use all labeled data for training and run inference on unlabeled data:
```
CUDA_VISIBLE_DEVICES=0 scripts/main.py --config-file configs/Bi-Matrix-V2.yaml \
--output results --mode inference
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/human_behavior_prediction/blob/master/LICENSE) file for details.
