# Reproducing "Mimetic Initialization of Self-Attention Layers"

CT-468 Deep Learning, paper reproduction. Team 404 Not Found: Syed Mohammad Huzaifa, Muhammad Maaz ur Rehman, Shaikh Muhammad Zain.

Paper: A. Trockman and J. Z. Kolter, Mimetic Initialization of Self-Attention Layers, ICML 2023, https://arxiv.org/abs/2305.09828

The paper has no official code. Everything here is written from scratch in plain PyTorch: the Vision Transformer, the mimetic initialization and the training loop. We reproduce the main CIFAR-10 result of Table 1 (ViT width 192, depth 12, 3 heads, 100 epochs): 86.07% with the default initialization and 90.78% with mimetic initialization.

## Files

- `code/model.py`: the ViT. Patch size 2 (256 patches plus a class token), fixed sinusoidal position embeddings, 12 pre-norm transformer blocks with separate query, key, value and projection layers, and a linear head on the class token.
- `code/mimetic_init.py`: `mimetic_init_(model)` sets the attention weights so that W_Q W_K^T is about 0.7 Z + 0.7 I in every head and W_V W_proj is about 0.4 Z - 0.4 I in every layer, using the SVD as in equations 1 and 2 of the paper. Nothing else in the model is changed.
- `code/train.py`: trains one model on CIFAR-10 and writes `log.csv` and `checkpoint.pt` to the output folder after every epoch.
- `code/kaggle_run.ipynb`, `code/kaggle_budget.ipynb`, `code/kaggle_posscale.ipynb`: the notebooks we ran on Kaggle (main runs, own experiment, position embedding scale test). Each one writes the three code files, runs the self-tests and then the training commands.
- `tools/make_kaggle_notebook.py`: builds those notebooks from the code files, so the notebooks always contain the current code.
- `tools/plot_results.py`: draws `results/test_acc.png` (test accuracy per epoch, 100-epoch runs) and `results/budget.png` (final accuracy at 25, 50 and 100 epochs) from the log files.
- `report/report.pdf`: the report.
- `slides/slides.pdf`: the slides for the viva.
- `results/`: our logs, one row per epoch (`base_log.csv` and `mimetic_log.csv` for the main runs, `{base,mimetic}_{25,50}_log.csv` for the own experiment, `mimetic_pos2_log.csv` for the position embedding test), the results tables `results.md` and the two plots.

## Requirements

- Python 3 with PyTorch and torchvision. We used PyTorch 2.11 and torchvision 0.26 on Colab and the preinstalled versions on Kaggle. matplotlib is needed only for the plot.
- An NVIDIA GPU with about 13 GB of memory for batch size 512. We used one Tesla T4 (15 GB); peak use was 12.6 GB.
- Internet access the first time: torchvision downloads CIFAR-10 (about 163 MB) into `./data`.

## How to re-run

### On Kaggle (what we did)

1. Create a new notebook and choose File > Import Notebook with one of the notebooks in `code/`.
2. Settings: Accelerator GPU T4 x2 (only one GPU is used), Internet on.
3. Save Version > Save & Run All (Commit). The browser can be closed. Logs and checkpoints appear under `runs/` in the version's Output tab.

### From the command line

Run these from inside `code/`:

```
python model.py            # self-test of the model
python mimetic_init.py     # self-test of the initialization
python train.py --init base    --out runs/base
python train.py --init mimetic --out runs/mimetic
```

Other options of `train.py`, with the defaults we used for the main runs: `--epochs 100 --warmup 5 --batch 512 --lr 3e-3 --wd 0.01 --seed 0 --pos-scale 1`. The model size options are `--dim 192 --depth 12 --heads 3`.

Own experiment (training budget) and the position embedding test:

```
python train.py --init mimetic --epochs 25 --out runs/mimetic_25
python train.py --init base    --epochs 25 --out runs/base_25
python train.py --init mimetic --epochs 50 --out runs/mimetic_50
python train.py --init base    --epochs 50 --out runs/base_50
python train.py --init mimetic --pos-scale 2 --out runs/mimetic_pos2
```

If a run stops (for example a Colab disconnect), run the same command again. It loads `checkpoint.pt` from the output folder and continues from the next epoch.

### Expected self-test output

`python model.py` should print about 5.34M parameters, output shape (2, 10), a maximum difference to hand-written attention of about 1e-6 or less, and `pos_scale 2 ratio: 2.0 2.0`.

`python mimetic_init.py` should print, for each of the 3 heads, a W_Q W_K^T diagonal mean of about +0.39 (below 0.7 because each head keeps only 64 of the 192 singular values) with off-diagonal mean |x| about 0.04, and a W_V W_proj diagonal mean of about -0.40 with off-diagonal mean |x| about 0.02.

### Run time

One epoch takes about 82 seconds on a T4, so a 100-epoch run takes about 2.3 hours. The first CIFAR-10 download on Kaggle took about 30 minutes.

## Training settings

From the paper (Section 5.1): AdamW, learning rate 3e-3, weight decay 0.01, batch size 512, 100 epochs, RandAugment and Cutout, alpha1 = beta1 = 0.7 and alpha2 = beta2 = 0.4, class token and sinusoidal position embeddings.

Not stated in the paper, chosen by us:

- 5 epochs of linear warmup, then cosine decay to 0, updated every step
- RandAugment with 2 operations at magnitude 9, Cutout with one 16 x 16 square
- random crop with 4-pixel padding and horizontal flip before RandAugment
- MLP ratio 4, GELU, no dropout, no label smoothing, no gradient clipping, weight decay on all parameters
- PyTorch default initialization for every layer in the base model, and for every layer except the attention weights in the mimetic model
- float16 mixed precision with a gradient scaler
- position embedding scale 1 for the main runs

We report test accuracy at the last epoch, fixed before training, and never the best epoch. There is no separate validation set.

## Results

Main runs (epoch 100, CIFAR-10 test set, one run each):

- Default initialization: 85.13% (paper 86.07%)
- Mimetic initialization: 88.87% (paper 90.78%)
- Gain: +3.74 points (paper +4.71)

Own experiment (separate runs of 25 and 50 epochs): the gain from mimetic init grows with training length, +1.82 at 25 epochs, +2.42 at 50 and +3.74 at 100. Position embedding scale 2 gave 88.59%, no better than scale 1.

Per-epoch numbers, the training curves and the results of the extra runs are in `results/results.md`. The report explains the gap to the paper.

## Code and tools

We used no code from other repositories. PyTorch and torchvision provide the layers, the optimizer, mixed precision, `F.scaled_dot_product_attention`, the CIFAR-10 loader and RandAugment. Cutout, the learning-rate schedule, the ViT and the mimetic initialization are written by us.

We used Claude (Anthropic), an AI assistant, as a helper during this project. We used it to explain parts of the paper and check our reading of its equations, to suggest and debug PyTorch code, to help set up and monitor the Colab and Kaggle runs, and to help draft and edit the notes, the report and this README. The team chose the paper, decided the experimental setup (including the settings the paper does not state and our own experiment), checked the results against the training logs, and reviewed and edited all code and text before submission. Every team member has gone through the code line by line and can explain it.
