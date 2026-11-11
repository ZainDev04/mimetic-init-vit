# Mimetic initialization (Trockman and Kolter, ICML 2023), plain PyTorch.
# Sets the attention weights so that, from step 0:
#   W_Q W_K^T  is about  alpha1 * Z1 + beta1 * I   (per head; tokens attend to themselves)
#   W_V W_proj is about  alpha2 * Z2 - beta2 * I   (whole layer; negative diagonal)
# Z1, Z2 have entries drawn from N(0, 1/d). We follow equations 1 and 2 of the paper,
# not equations 3 to 7, which mix up the subscripts (see the report, section 1).
#
# Reminder: nn.Linear stores weight as (out, in) and computes x @ weight.T,
# so the paper's W_Q is q.weight.T, and setting W_Q = A means q.weight = A.T.

import torch
from model import Attention, ViT


@torch.no_grad()   # we are writing weights by hand, so no gradients are needed
def mimetic_init_(model, alpha1=0.7, beta1=0.7, alpha2=0.4, beta2=0.4):
    for m in model.modules():
        if not isinstance(m, Attention):
            continue
        d = m.q.weight.shape[1]   # embedding dim, 192
        k = m.head_dim            # dim per head, 64
        I = torch.eye(d)

        # Query-key: a new random target M1 for every head.
        for h in range(m.heads):
            rows = slice(h * k, (h + 1) * k)   # the output units that belong to head h
            M1 = alpha1 * torch.randn(d, d) / d ** 0.5 + beta1 * I
            U, S, Vh = torch.linalg.svd(M1)    # M1 = U diag(S) Vh, with Vh = V^T
            # Each head has only k columns, so keep the top k singular values.
            # W_Q = U[:, :k] sqrt(S), W_K = V[:, :k] sqrt(S), so W_Q W_K^T = best rank-k copy of M1.
            s = S[:k].sqrt()
            m.q.weight[rows] = (U[:, :k] * s).T   # (k, d)
            m.k.weight[rows] = s[:, None] * Vh[:k]  # (k, d), this is (V[:, :k] * s).T

        # Value-projection: one target M2 for the whole layer, full rank.
        M2 = alpha2 * torch.randn(d, d) / d ** 0.5 - beta2 * I
        U, S, Vh = torch.linalg.svd(M2)
        s = S.sqrt()
        # W_V = U sqrt(S), W_proj = sqrt(S) V^T, so W_V W_proj = M2 exactly.
        m.v.weight.copy_((U * s).T)
        m.proj.weight.copy_((s[:, None] * Vh).T)
    return model


if __name__ == "__main__":
    # Check on Colab with `python mimetic_init.py`.
    torch.manual_seed(0)
    model = mimetic_init_(ViT())
    attn = model.blocks[0].attn
    k = attn.head_dim

    def diag_report(name, P):
        d = P.shape[0]
        diag = P.diagonal().mean().item()
        off = (P - torch.diag(P.diagonal())).abs().sum().item() / (d * d - d)
        print(f"{name}: diagonal mean {diag:+.3f}, off-diagonal mean |x| {off:.3f}")

    for h in range(attn.heads):
        rows = slice(h * k, (h + 1) * k)
        WQ = attn.q.weight[rows].T             # (d, k)
        WK = attn.k.weight[rows].T             # (d, k)
        diag_report(f"head {h} W_Q W_K^T ", WQ @ WK.T)   # expect clearly positive diagonal
    WV = attn.v.weight.T
    WP = attn.proj.weight.T
    diag_report("layer   W_V W_proj", WV @ WP)            # expect about -0.4 on the diagonal
