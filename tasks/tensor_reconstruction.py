'''
Run tensor reconstruction experiments and save reconstruction artifacts.

This module loads irregular tensor data, trains a selected PARAFAC2 model,
evaluates reconstruction error, and optionally saves reconstructed tensors,
singular values, training curves, and diagnostic plots.
'''

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from data import read_completion_data
from model import *
from train import *
from utils import *


def save_reconstruction_tensors(
    X,
    X_hat,
    train_mask,
    save_path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    '''
    Save the original slices, reconstructed slices, masks, and metadata.

    Inputs:
        X: Original tensor slices.
        X_hat: Reconstructed tensor slices.
        train_mask: Observation masks used for training.
        save_path: Path of the output PyTorch file.
        extra: Optional experiment metadata.

    Output:
        None. A PyTorch file is written to save_path.
    '''
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def detach_cpu(obj):
        '''
        Detach tensors and move them to the CPU.
        '''
        if isinstance(obj, (list, tuple)):
            return [x.detach().cpu() for x in obj]

        return obj.detach().cpu()

    payload = {
        "X": detach_cpu(X),
        "X_hat": detach_cpu(X_hat),
        "train_mask": detach_cpu(train_mask),
        "extra": extra or {},
    }

    torch.save(payload, save_path)

    print(f"[Reconstruction] saved X and X_hat to {save_path}")


def save_slice_singular_values(
    X_hat,
    save_path,
    center: bool = False,
    normalize: bool = False,
) -> None:
    '''
    Compute and save singular values for each reconstructed slice.

    Inputs:
        X_hat: Reconstructed tensor slices with shape (I_k, J).
        save_path: Path of the output NumPy file.
        center: Whether to subtract the slice mean before SVD.
        normalize: Whether to divide by the largest singular value.

    Output:
        None. A NumPy NPZ file is written to save_path.
    '''
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    singular_values = []

    with torch.no_grad():
        for Xk_hat in X_hat:
            Xk_hat = Xk_hat.detach()

            if center:
                Xk_hat = Xk_hat - Xk_hat.mean()

            values = torch.linalg.svdvals(Xk_hat)

            if normalize:
                values = values / values[0].clamp_min(1e-12)

            singular_values.append(values.cpu().numpy())

    # Slice matrices may have different dimensions.
    singular_values = np.array(singular_values, dtype=object)

    np.savez(
        save_path,
        singular_values=singular_values,
        center=center,
        normalize=normalize,
    )

    print(f"[SVD] saved singular values to {save_path}")


def run_reconstruction(args, device) -> Optional[float]:
    '''
    Train and evaluate models for an irregular tensor reconstruction task.

    Inputs:
        args: Parsed experiment arguments.
        device: PyTorch device used for training and evaluation.

    Output:
        The relative Frobenius error of the final Rank experiment.
    '''
    X, train_mask, _, _, _, _, length, max_length = read_completion_data(
        data_name=args.data,
        missing_ratio=args.missing_ratio,
        seed=args.seed,
        device=device,
    )

    if len(X) == 0:
        raise ValueError("The loaded tensor contains no slices.")

    J = X[0].shape[1]
    R_list = parse_int_list(args.R_list) or [int(args.R)]

    if any(R <= 0 for R in R_list):
        raise ValueError(f"All ranks must be positive: {R_list}")

    if isinstance(train_mask, (list, tuple)):
        nnz = sum(int(mask.sum().item()) for mask in train_mask)
    else:
        nnz = int(train_mask.sum().item())

    print("================ Reconstruction Task ================")
    print(f"# slices: {len(X)}")
    print(f"J: {J}")
    print(f"Rank: {R_list}")
    print(f"max length: {max_length}")
    print(f"nnz: {nnz:,}")
    print(f"first slice shape: {X[0].shape}")
    print(f"model: {args.model}")
    print(f"trainer: {args.trainer}")

    last_nre = None
    true_ranks = slice_true_ranks(X)

    for R in R_list:
        num_masks = int(round(float(R * args.mask_ratio)))

        print("-----------------------------------------------------")
        print(f"[Reconstruction] Start R={R}")

        if args.model == "parafac2":
            if args.trainer != "als":
                raise ValueError(
                    f"Unknown trainer for PARAFAC2: {args.trainer}"
                )

            model = PARAFAC2_ALS(
                length=length,
                J=J,
                R=R,
                device=device,
            ).to(device)

        elif args.model == "pmask":
            if num_masks <= 0:
                raise ValueError(
                    "PMask requires at least one mask. "
                    f"Received R={R} and mask_ratio={args.mask_ratio}."
                )

            base_rank = R - num_masks

            if base_rank <= 0:
                raise ValueError(
                    "PMask base Rank must be positive. "
                    f"Received R={R} and num_masks={num_masks}."
                )

            model = PARAFAC2_PMask(
                length=length,
                J=J,
                R=base_rank,
                L=num_masks,
                device=device,
                init_scale=0.5,
            ).to(device)

        elif args.model == "nomask":
            model = PARAFAC2_NoMask(
                length=length,
                J=J,
                R=R,
                device=device,
                init_scale=0.5,
            ).to(device)

        else:
            raise ValueError(f"Unknown model: {args.model}")

        trainer = get_trainer(args)

        save_path = None

        if (
            args.trainer == "gd"
            and args.save_dir
            and not args.skip_artifacts
        ):
            ratio = str(args.missing_ratio).replace(".", "p")

            save_path = (
                Path(args.save_dir)
                / "reconstruction"
                / (
                    f"{args.data}_{args.model}_{args.trainer}_"
                    f"R{R}_missing{ratio}_seed{args.seed}.pt"
                )
            )

        curve_path = None

        if not args.skip_artifacts:
            curve_path = (
                f"results/reconstruction_"
                f"{args.data}_{args.model}_{args.trainer}_R{R}_curve.csv"
            )

        trainer.fit(
            model=model,
            X=X,
            train_mask=train_mask,
            save_path=save_path,
            save_meta={
                "task": "reconstruction",
                "data": args.data,
                "model": args.model,
                "trainer": args.trainer,
                "R": R,
                "effective_R": R,
                "base_R": R - num_masks if args.model == "pmask" else R,
                "num_masks": num_masks if args.model == "pmask" else 0,
                "missing_ratio": args.missing_ratio,
                "mask_ratio": args.mask_ratio,
                "seed": args.seed,
                "lr": args.lr,
                "l2": args.l2,
                "lambda_align": args.uniqueness,
                "lambda_div": args.diversity,
            },
            curve_path=curve_path,
        )

        with torch.no_grad():
            X_hat = model.reconstruct()

            nre = reconstruction_nre(
                X_true=X,
                X_pred=X_hat,
                mask=train_mask,
            )

            err_info = reconstruction_relative_fro_error(
                X_true=X,
                X_pred=X_hat,
                mask=train_mask,
            )

        last_nre = float(nre)

        if not args.skip_artifacts:
            plot_rank_vs_rel_fro(
                true_ranks=true_ranks,
                slice_rel_fro=err_info["slice_rel_fro"],
                global_rank=R,
                title="Relationship between Rank and NRE",
                ylabel="NRE",
                out_path=(
                    f"checkpoints/plots/reconstruction_"
                    f"{args.data}_{args.model}_R{R}_rank_vs_nre.png"
                ),
            )

        print(
            f"[Reconstruction] R={R}, "
            f"Relative Frobenius Error: {last_nre:.6f}"
        )

        if not args.skip_artifacts:
            save_slice_singular_values(
                X_hat=X_hat,
                save_path=(
                    f"checkpoints/svd/"
                    f"{args.data}_{args.model}_R{R}_"
                    f"{args.mask_ratio}_singular_values.npz"
                ),
                center=False,
                normalize=False,
            )

            save_reconstruction_tensors(
                X=X,
                X_hat=X_hat,
                train_mask=train_mask,
                save_path=(
                    f"checkpoints/reconstruction_tensors/"
                    f"{args.data}_{args.model}_R{R}_"
                    f"{args.mask_ratio}_seed{args.seed}.pt"
                ),
                extra={
                    "data": args.data,
                    "model": args.model,
                    "trainer": args.trainer,
                    "R": R,
                    "effective_R": R,
                    "base_R": (
                        R - num_masks
                        if args.model == "pmask"
                        else R
                    ),
                    "num_masks": (
                        num_masks
                        if args.model == "pmask"
                        else 0
                    ),
                    "mask_ratio": args.mask_ratio,
                    "missing_ratio": args.missing_ratio,
                    "seed": args.seed,
                    "nre": last_nre,
                },
            )

    return last_nre

