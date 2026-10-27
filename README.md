# Lab - Setup a project from scratch

Training of a small custom CNN on Tiny-ImageNet (200 classes).

## Structure

- `models/custom_net.py`: the network
- `dataset/tiny_imagenet.py`: transforms and dataloaders
- `utils/prepare_tiny_imagenet.py`: downloads the dataset into `data/` and reorganizes the val split
- `train.py`: training loop, saves the best weights in `checkpoints/best.pth`
- `eval.py`: validation loop, can evaluate a saved checkpoint

## Run on Colab

```
!git clone https://github.com/frastud812/aml-lab3.git
%cd aml-lab3
!pip install -r requirements.txt
!python utils/prepare_tiny_imagenet.py
!python train.py --epochs 10
```

To log the metrics on Weights & Biases, run `wandb login` first and add `--wandb`.

To evaluate the saved model: `!python eval.py --checkpoint checkpoints/best.pth`
