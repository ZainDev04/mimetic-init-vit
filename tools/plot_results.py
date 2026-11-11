"""Plot the results of all runs.

Usage: python tools/plot_results.py
Reads results/*_log.csv and writes two figures:
  results/test_acc.png: test accuracy per epoch, 100-epoch runs (base, mimetic, mimetic with pos scale 2)
  results/budget.png:   final test accuracy against training length (25, 50, 100 epochs)
"""
import csv
import matplotlib.pyplot as plt

PAPER = {"base": 86.07, "mimetic": 90.78}  # Table 1, ViT 192/12/3, CIFAR-10
COLORS = {"base": "tab:blue", "mimetic": "tab:orange", "mimetic_pos2": "tab:green"}


def test_acc(name):
    with open(f"results/{name}_log.csv") as f:
        return [float(r["test_acc"]) for r in csv.DictReader(f)]


# Figure 1: test accuracy per epoch of the 100-epoch runs
for name in ["base", "mimetic", "mimetic_pos2"]:
    acc = test_acc(name)
    plt.plot(range(1, len(acc) + 1), acc, color=COLORS[name], label=f"{name} (ours, final {acc[-1]:.2f})")
for name in ["base", "mimetic"]:
    plt.axhline(PAPER[name], color=COLORS[name], linestyle="--", label=f"{name} (paper, {PAPER[name]})")
plt.xlabel("epoch")
plt.ylabel("test accuracy (%)")
plt.title("ViT 192/12/3 on CIFAR-10")
plt.ylim(30, 95)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("results/test_acc.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved results/test_acc.png")

# Figure 2: final test accuracy of separate 25, 50 and 100 epoch runs
budgets = [25, 50, 100]
for init in ["base", "mimetic"]:
    # the 100-epoch runs are the main runs, saved without a suffix
    finals = [test_acc(f"{init}_{b}" if b != 100 else init)[-1] for b in budgets]
    plt.plot(budgets, finals, "o-", color=COLORS[init], label=init)
    for b, a in zip(budgets, finals):
        plt.annotate(f"{a:.2f}", (b, a), textcoords="offset points", xytext=(0, 8), ha="center")
plt.xlabel("training epochs (each a separate run with its own schedule)")
plt.ylabel("final test accuracy (%)")
plt.title("Training budget: base vs mimetic")
plt.xticks(budgets)
plt.ylim(76, 91)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("results/budget.png", dpi=150, bbox_inches="tight")
print("saved results/budget.png")
