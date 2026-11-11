# Build a Kaggle notebook from the three code files, for "Save & Run All" on Kaggle.
# Usage: python tools/make_kaggle_notebook.py          -> code/kaggle_run.ipynb (100-epoch main runs)
#        python tools/make_kaggle_notebook.py budget   -> code/kaggle_budget.ipynb (own experiment: 25 and 50 epochs)
#        python tools/make_kaggle_notebook.py posscale -> code/kaggle_posscale.ipynb (gap test: mimetic, gamma = 2)
import json
import os
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "main"

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


cells = [cell("# Work in /kaggle/temp so the dataset is not saved as output; only runs/ goes to /kaggle/working.\n"
              "!mkdir -p /kaggle/temp\n"
              "%cd /kaggle/temp")]
for name in ["model.py", "mimetic_init.py", "train.py"]:
    with open(os.path.join(root, "code", name), encoding="utf-8") as f:
        cells.append(cell("%%writefile " + name + "\n" + f.read()))
cells.append(cell("# Check the GPU and run the self-tests.\n"
                  "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
                  "!python model.py\n"
                  "!python mimetic_init.py"))
if mode == "posscale":
    # Gap test: the main mimetic run again, with position embeddings multiplied by 2 (gamma in Fig. 4).
    cells.append(cell("# Gap test: mimetic, 100 epochs, position embedding scale 2.\n"
                      "!python train.py --init mimetic --pos-scale 2 --out /kaggle/working/runs/mimetic_pos2"))
    out_name = "kaggle_posscale.ipynb"
elif mode == "budget":
    # Own experiment: same settings as the main runs, only --epochs changes (the cosine schedule
    # then ends at that epoch). Warmup stays 5 epochs. Shortest runs first, about 3.5 h in total.
    cells.append(cell("# Own experiment: base vs mimetic with 25 and 50 epochs of training.\n"
                      "!python train.py --init mimetic --epochs 25 --out /kaggle/working/runs/mimetic_25\n"
                      "!python train.py --init base --epochs 25 --out /kaggle/working/runs/base_25\n"
                      "!python train.py --init mimetic --epochs 50 --out /kaggle/working/runs/mimetic_50\n"
                      "!python train.py --init base --epochs 50 --out /kaggle/working/runs/base_50"))
    out_name = "kaggle_budget.ipynb"
else:
    cells.append(cell("# Full runs: 100 epochs each. Logs and checkpoints end up in the notebook's Output tab.\n"
                      "!python train.py --init mimetic --out /kaggle/working/runs/mimetic\n"
                      "!python train.py --init base --out /kaggle/working/runs/base"))
    out_name = "kaggle_run.ipynb"

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(os.path.join(root, "code", out_name), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote code/" + out_name + " with", len(cells), "cells")
