import random
import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def kron(A, B):
    """
    Kronecker product.

    A: I x R
    B: J x R

    return:
        IJ x R
    """

    return torch.einsum("ab,cd->acbd", A, B).reshape(
        A.size(0) * B.size(0),
        A.size(1) * B.size(1),
    )


def khatri_rao(A, B):
    """
    Column-wise Kronecker product.

    A: I x R
    B: J x R

    return:
        IJ x R
    """

    return torch.cat(
        [
            kron(A[:, r].unsqueeze(1), B[:, r].unsqueeze(1))
            for r in range(A.size(1))
        ],
        dim=1,
    )



def generate_irregular_tensor(
    num_slices=16,
    row_choices=(512, 1024),
    num_columns=256,
    seed=42,
    normalize="fro",
    min_rank=64,
    max_rank=256,
    noise_scale=0.0,
):
    random.seed(seed)
    np.random.seed(seed)

    tensor = []

    # Set slice ranks as an arithmetic sequence.
    ranks = np.linspace(min_rank, max_rank, num_slices)
    ranks = np.round(ranks).astype(int)

    for k in range(num_slices):
        rows = random.choice(row_choices)

        rank_k = ranks[k]
        rank_k = min(rank_k, rows, num_columns)

        # Low-rank matrix: X_k = A_k @ B_k.T
        A_k = np.random.randn(rows, rank_k)
        B_k = np.random.randn(num_columns, rank_k)

        X_k = A_k @ B_k.T

        if noise_scale is not None and noise_scale > 0:
            X_k = X_k + noise_scale * np.random.randn(rows, num_columns)

        if normalize == "fro":
            X_k = X_k - X_k.mean()

            fro = np.linalg.norm(X_k, ord="fro")
            X_k = X_k / max(fro, 1e-12) * np.sqrt(rows * num_columns)

        elif normalize == "col":
            col_norm = np.linalg.norm(X_k, axis=0, keepdims=True)
            X_k = X_k / np.maximum(col_norm, 1e-12)

        elif normalize is None:
            pass

        else:
            raise ValueError(f"Unknown normalize: {normalize}")

        tensor.append(X_k.astype(np.float32))

        numerical_rank = np.linalg.matrix_rank(X_k)

        print(
            f"slice {k:02d} | "
            f"shape: {X_k.shape} | "
            f"target rank: {rank_k:03d} | "
            f"numerical rank: {numerical_rank:03d} | "
            f"mean: {X_k.mean():.6f} | "
            f"std: {X_k.std():.6f} | "
            f"fro: {np.linalg.norm(X_k, ord='fro'):.6f}"
        )

    return tensor    

def completion_rmse(X_pred, test_idx, test_val, eps=1e-12):
    """
    Completion RMSE over held-out entries.

    RMSE = sqrt(
        sum_k ||P_test(X_k - X_hat_k)||_F^2
        /
        total number of held-out entries
    )
    """

    device = X_pred[0].device
    dtype = X_pred[0].dtype

    squared_error_sum = torch.zeros((), device=device, dtype=dtype)
    count = torch.zeros((), device=device, dtype=dtype)

    for k in range(len(test_idx)):
        if len(test_idx[k]) == 0:
            continue

        indices = torch.as_tensor(
            test_idx[k],
            dtype=torch.long,
            device=device,
        )

        i_idx = indices[:, 0]
        j_idx = indices[:, 1]

        pred_vals = X_pred[k][i_idx, j_idx]
        true_vals = test_val[k].to(device=device, dtype=dtype)

        diff = pred_vals - true_vals

        squared_error_sum = squared_error_sum + torch.sum(diff ** 2)
        count = count + diff.numel()

    if count.item() == 0:
        return float("inf")

    return torch.sqrt(squared_error_sum / count.clamp_min(eps))

def completion_nre(X_pred, test_idx, test_val, eps=1e-12):
    """
    Completion relative Frobenius norm over held-out entries.

    NRE = ||P_test(X - X_hat)||_F / ||P_test(X)||_F

    This matches relative_fro_error style:
        sqrt(sum_k ||P_test(X_k - X_hat_k)||_F^2)
        /
        sqrt(sum_k ||P_test(X_k)||_F^2)
    """

    if len(X_pred) == 0:
        return float("inf")

    device = X_pred[0].device
    dtype = X_pred[0].dtype

    numerator_sq = torch.zeros((), device=device, dtype=dtype)
    denominator_sq = torch.zeros((), device=device, dtype=dtype)

    has_test = False

    for k in range(len(test_idx)):
        if len(test_idx[k]) == 0:
            continue

        has_test = True
        device_k = X_pred[k].device
        dtype_k = X_pred[k].dtype

        indices = torch.as_tensor(
            test_idx[k],
            dtype=torch.long,
            device=device_k,
        )

        i_idx = indices[:, 0]
        j_idx = indices[:, 1]

        pred_vals = X_pred[k][i_idx, j_idx]
        true_vals = test_val[k].to(device=device_k, dtype=dtype_k)

        diff = pred_vals - true_vals

        numerator_sq = numerator_sq.to(device=device_k, dtype=dtype_k) + torch.sum(diff ** 2)
        denominator_sq = denominator_sq.to(device=device_k, dtype=dtype_k) + torch.sum(true_vals ** 2)

    if not has_test:
        return float("inf")

    return torch.sqrt(numerator_sq / denominator_sq.clamp_min(eps))

def reconstruction_nre(X_true, X_pred, mask, eps=1e-12):
    """
    Reconstruction relative Frobenius norm over evaluated entries.

    NRE = ||P_obs(X - X_hat)||_F / ||P_obs(X)||_F

    This matches relative_fro_error style:
        sqrt(sum_k ||P_obs(X_k - X_hat_k)||_F^2)
        /
        sqrt(sum_k ||P_obs(X_k)||_F^2)
    """

    device = X_pred[0].device
    dtype = X_pred[0].dtype

    numerator_sq = torch.zeros((), device=device, dtype=dtype)
    denominator_sq = torch.zeros((), device=device, dtype=dtype)

    for k in range(len(X_true)):
        mask_k = mask[k].to(device=device, dtype=torch.bool)

        true_vals = X_true[k].to(device=device, dtype=dtype)[mask_k]
        pred_vals = X_pred[k].to(device=device, dtype=dtype)[mask_k]

        diff = pred_vals - true_vals

        numerator_sq = numerator_sq + torch.sum(diff ** 2)
        denominator_sq = denominator_sq + torch.sum(true_vals ** 2)

    return torch.sqrt(numerator_sq / denominator_sq.clamp_min(eps))


@torch.no_grad()
def plot_slice_svd_spectrum(
    X_hat,
    slice_idx=0,
    rank_budget=None,
    title_prefix="Reconstructed Slice",
    save_path=None,
    show=True,
):
    """
    Plot singular value spectrum of one reconstructed slice.
    """
    Xk = X_hat[slice_idx]

    if isinstance(Xk, torch.Tensor):
        Xk = Xk.detach().float().cpu()
    else:
        Xk = torch.tensor(Xk, dtype=torch.float32)

    s = torch.linalg.svdvals(Xk).numpy()

    plt.figure(figsize=(6.5, 4.0))
    plt.plot(
        range(1, len(s) + 1),
        s,
        marker="o",
        linewidth=2.0,
        markersize=3.5,
        label="Singular values",
    )

    if rank_budget is not None:
        plt.axvline(
            rank_budget,
            linestyle="--",
            linewidth=1.5,
            label=f"Rank budget = {rank_budget}",
        )

    plt.xlabel("Singular value index")
    plt.ylabel("Singular value")
    plt.title(f"{title_prefix} (slice {slice_idx})")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

    return s

def reconstruction_relative_fro_error(X_true, X_pred, mask, eps=1e-12):
    """
    Reconstruction relative Frobenius error over evaluated entries.

    Global relative Frobenius error:
        sqrt(sum_k ||P_obs(X_k - Xhat_k)||_F^2)
        /
        sqrt(sum_k ||P_obs(X_k)||_F^2)

    Slice-wise relative Frobenius error:
        ||P_obs(X_k - Xhat_k)||_F
        /
        ||P_obs(X_k)||_F
    """

    device = X_pred[0].device
    dtype = X_pred[0].dtype

    global_num_sq = torch.zeros((), device=device, dtype=dtype)
    global_den_sq = torch.zeros((), device=device, dtype=dtype)

    slice_rel_fro = []

    for k in range(len(X_true)):
        mask_k = mask[k].to(device=device, dtype=torch.bool)

        true_vals = X_true[k].to(device=device, dtype=dtype)[mask_k]
        pred_vals = X_pred[k].to(device=device, dtype=dtype)[mask_k]

        diff = pred_vals - true_vals

        num_k = torch.sum(diff ** 2)
        den_k = torch.sum(true_vals ** 2)

        rel_fro_k = torch.sqrt(num_k / den_k.clamp_min(eps))
        slice_rel_fro.append(rel_fro_k)

        global_num_sq = global_num_sq + num_k
        global_den_sq = global_den_sq + den_k

    global_rel_fro = torch.sqrt(global_num_sq / global_den_sq.clamp_min(eps))

    return {
        "global_rel_fro": global_rel_fro,
        "slice_rel_fro": torch.stack(slice_rel_fro),
    }


@torch.no_grad()
def slice_true_ranks(X_true, energy_threshold=0.99, eps=1e-12):
    """
    Compute effective rank for each slice using SVD energy.

    Effective rank q_k is the smallest q such that:

        sum_{r=1}^q sigma_r^2 / sum_r sigma_r^2 >= energy_threshold

    Args:
        X_true:
            list of tensors, X_true[k] has shape (I_k, J)

        energy_threshold:
            Energy threshold, e.g., 0.95.

    Returns:
        ranks:
            torch.LongTensor of shape (K,)
    """

    ranks = []

    for X_k in X_true:
        X_k = X_k.detach().float().cpu()

        s = torch.linalg.svdvals(X_k)
        energy = s ** 2
        total_energy = energy.sum()

        if total_energy <= eps:
            rank_k = torch.tensor(0, dtype=torch.long)
        else:
            cumulative = torch.cumsum(energy, dim=0) / total_energy.clamp_min(eps)

            # first index where cumulative >= threshold
            rank_k = torch.searchsorted(
                cumulative,
                torch.tensor(float(energy_threshold), dtype=cumulative.dtype),
                right=False,
            ) + 1

            rank_k = rank_k.to(torch.long)

        ranks.append(rank_k)

    return torch.stack(ranks)

@torch.no_grad()
def plot_rank_vs_rel_fro(
    true_ranks,
    slice_rel_fro,
    global_rank=None,
    title="Relationship between Rank and Relative Frobenius Error",
    xlabel="True Rank",
    ylabel="Relative Frobenius Error",
    out_path="rank_vs_rel_fro.png",
):
    """
    Scatter plot between true slice rank and slice-wise relative Frobenius error.

    Args:
        true_ranks:
            Tensor/list of shape (K,), true matrix rank for each slice.

        slice_rel_fro:
            Tensor/list of shape (K,), slice-wise relative Frobenius error.

        global_rank:
            Optional vertical reference line, e.g., model rank R.

        out_path:
            Path to save figure.

    Returns:
        stats dict with Pearson r, R^2, slope, intercept, and saved path.
    """

    x = torch.as_tensor(true_ranks, dtype=torch.float32).detach().cpu()
    y = torch.as_tensor(slice_rel_fro, dtype=torch.float32).detach().cpu()

    valid = torch.isfinite(x) & torch.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.numel() < 2:
        raise ValueError("Need at least two valid points to plot rank vs relative Frobenius error.")

    x_mean = x.mean()
    y_mean = y.mean()

    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2)).clamp_min(1e-12)
    pearson_r = torch.sum(x_centered * y_centered) / denom

    slope = torch.sum(x_centered * y_centered) / torch.sum(x_centered ** 2).clamp_min(1e-12)
    intercept = y_mean - slope * x_mean

    y_fit = slope * x + intercept
    ss_res = torch.sum((y - y_fit) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    order = torch.argsort(x)
    x_line = x[order]
    y_line = y_fit[order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.scatter(x.numpy(), y.numpy(), alpha=0.35, label="Slices")
    ax.plot(x_line.numpy(), y_line.numpy(), linewidth=3, label="Linear fit")

    if global_rank is not None:
        ax.axvline(float(global_rank), linestyle="--", linewidth=2.5, label=f"Global rank = {global_rank}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    text = f"Pearson r = {pearson_r.item():.3f}\n$R^2$ = {r2.item():.3f}"
    ax.text(
        0.97,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "pearson_r": pearson_r.item(),
        "r2": r2.item(),
        "slope": slope.item(),
        "intercept": intercept.item(),
        "n_points": int(x.numel()),
        "out_path": str(out_path),
    }