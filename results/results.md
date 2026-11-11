# Results: ViT 192/12/3 on CIFAR-10

Runs: Kaggle, one Tesla T4, batch 512, finished 2026-10-30 (one notebook version, 18502 s for both runs). Per-epoch numbers are in `base_log.csv` and `mimetic_log.csv`. Plots: `test_acc.png` and `budget.png` (made by `python tools/plot_results.py`).

## Main table

Test accuracy (%) on the 10,000 CIFAR-10 test images. "Ours" is the epoch-100 value, fixed in advance (see the note at the end).

| Init | Paper (Table 1) | Ours | Ours minus paper |
|---|---|---|---|
| Base (PyTorch default) | 86.07 | 85.13 | -0.94 |
| Mimetic | 90.78 | 88.87 | -1.91 |
| Gain from mimetic | +4.71 | +3.74 | -0.97 |

The paper's claim holds: mimetic init beats the base init by a clear margin on the same model and training recipe. Our gain is 79% of the paper's gain (3.74 / 4.71).

## Accuracy during training

| Epoch | Base | Mimetic | Gain |
|---|---|---|---|
| 10 | 64.06 | 66.21 | +2.15 |
| 25 | 74.09 | 75.16 | +1.07 |
| 50 | 82.12 | 85.53 | +3.41 |
| 75 | 84.91 | 87.45 | +2.54 |
| 100 | 85.13 | 88.87 | +3.74 |

These epochs sit inside one 100-epoch cosine schedule, so they are not the same as separate 10, 25 or 50 epoch runs.

## Final training numbers (epoch 100)

- Base: train loss 0.167, train accuracy 94.13
- Mimetic: train loss 0.149, train accuracy 94.80

Mimetic is higher on both training and test accuracy, so it fits the training data better and also generalises better. The train-test gap is 9.00 points for base and 5.93 for mimetic.

## Note on reporting

There is no validation set (the paper does not describe one). Test accuracy was printed every epoch only to watch training. We decided before training to report epoch 100, not the best epoch, so the test set was not used to choose anything. Best epochs for reference: base 85.23 (epoch 89), mimetic 88.93 (epoch 98).

## Own experiment: training budget

Question: does the gain from mimetic init depend on how long we train? Base and mimetic were each trained as separate runs of 25 and 50 epochs, each with its own cosine schedule. Only `--epochs` changed; warmup stayed 5 epochs and the seed stayed 0. The 100-epoch numbers are the main runs. Runs: one Kaggle notebook version on a T4, 14,709 s for the four runs. Per-epoch numbers: `{base,mimetic}_{25,50}_log.csv`.

| Epochs | Base | Mimetic | Gain |
|---|---|---|---|
| 25 | 79.21 | 81.03 | +1.82 |
| 50 | 83.98 | 86.40 | +2.42 |
| 100 | 85.13 | 88.87 | +3.74 |

Test accuracy at the last epoch of each run.

What we see:

- Mimetic is better at every budget, and the gain grows with the budget: +1.82, +2.42, +3.74.
- Mimetic at 50 epochs (86.40) is higher than base at 100 epochs (85.13), so mimetic reaches the base model's final accuracy in half the training.
- Going from 50 to 100 epochs adds 1.15 points for base but 2.47 for mimetic. The base model is close to its limit at 50 epochs; the mimetic model keeps improving.
- A short run with its own schedule ends far higher than the same epoch inside a 100-epoch run (mimetic 81.03 against 75.16 at epoch 25, base 79.21 against 74.09), because the learning rate has decayed to 0 by the end.

Limits: one seed per setting. The pos-scale run below differs from the main mimetic run by 0.28 points, which gives a rough idea of the noise between runs. The rise from +1.82 to +3.74 is much larger than that, but the step from 25 to 50 epochs (+0.60) is not far above it.

## Gap test: position embedding scale 2

One mimetic run with the sinusoidal position embeddings multiplied by 2 (`--pos-scale 2`), everything else as in the main mimetic run. Run: Kaggle, T4, 9,510 s. Per-epoch numbers: `mimetic_pos2_log.csv`, also in `test_acc.png`.

- Epoch 100: 88.59, against 88.87 with scale 1 (-0.28). Best epoch 88.69 (epoch 93).
- Train accuracy 94.39, train loss 0.161 (scale 1: 94.80, 0.149).

Scale 2 did not help in our setup, so it does not explain the gap to the paper's 90.78. See section 5 of the report.

