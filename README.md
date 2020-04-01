# Human Strategic Behavior Prediction

## Training

```
CUDA_VISIBLE_DEVICES=0 python -u main.py --num-kernels 64
```

## Inference

```
CUDA_VISIBLE_DEVICES=0 python -u main.py --num-kernels 64 --inference
```