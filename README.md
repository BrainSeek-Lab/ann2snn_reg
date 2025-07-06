# ANN_SNN_QCFS

## Training

```bash
PYTHONPATH="" python main_train.py \
  -data cifar10 -arch vgg16 \
  -L 8 \
  -dev 0 
```

## Testing

Evaluate the converted SNN:

```bash
PYTHONPATH="" python main_test.py -id="vgg16_L[8]" -data=cifar10 -T=4 -dev=0 -
arch=vgg16
```

Results (checkpoint and logs) appear in `<dataset>-checkpoints/`.
Feel free to adjust `-L`, `-T`.

