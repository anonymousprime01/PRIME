'''
Run tensor completion experiments on irregular tensor data.

This module loads incomplete tensor slices, trains the selected PARAFAC2 model,
and evaluates predictions on held-out test entries.
'''

from pathlib import Path
from typing import Optional

import torch

from data import read_completion_data
from model import *
from train import get_trainer
from utils import completion_rmse, parse_int_list


def run_completion(args, device) -> Optional[float]:
    '''
    Train and evaluate models for an irregular tensor completion task.

    Inputs:
        args: Parsed experiment arguments.
        device: PyTorch device used for training and evaluation.

    Output:
        The held-out test RMSE of the final Rank experiment.
    '''
    if args.missing_ratio <= 0:
        raise ValueError(
            "Completion requires --missing_ratio > 0 because validation "
            "and test entries are required. Use --task reconstruction "
            "when missing_ratio is 0."
        )

    (
        X,
        train_mask,
        valid_idx,
        valid_val,
        test_idx,
        test_val,
        length,
        max_length,
    ) = read_completion_data(
        data_name=args.data,
        missing_ratio=args.missing_ratio,
        seed=args.seed,
        device=device,
    )

    if len(X) == 0:
        raise ValueError("The loaded tensor contains no slices.")

    J = X[0].shape[1]

    total_entries = sum(x.numel() for x in X)
    total_rows = sum(x.shape[0] for x in X)

    R_list = parse_int_list(args.R_list) or [int(args.R)]

    if any(R <= 0 for R in R_list):
        raise ValueError(f"All ranks must be positive: {R_list}")

    print("================ Completion Task ================")
    print(f"# slices: {len(X)}")
    print(f"J: {J}")
    print(f"total rows: {total_rows}")
    print(f"total entries: {total_entries}")
    print(f"Rank list: {R_list}")
    print(f"missing ratio: {args.missing_ratio}")
    print(f"max length: {max_length}")
    print(f"first slice shape: {X[0].shape}")
    print(f"model: {args.model}")
    print(f"trainer: {args.trainer}")

    last_rmse = None

    for R in R_list:
        num_masks = int(round(float(R * args.mask_ratio)))

        print("-----------------------------------------------------")
        print(f"[Completion] Start R={R}")

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
                init_scale=1.0,
            ).to(device)

        elif args.model == "nomask":
            model = PARAFAC2_NoMask(
                length=length,
                J=J,
                R=R,
                device=device,
                init_scale=1.0,
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
                / "completion"
                / (
                    f"{args.data}_{args.model}_{args.trainer}_"
                    f"R{R}_missing{ratio}_seed{args.seed}.pt"
                )
            )

        trainer.fit(
            model=model,
            X=X,
            train_mask=train_mask,
            valid_mask=(valid_idx, valid_val),
            save_path=save_path,
            save_meta={
                "task": "completion",
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
                "lr": args.lr,
                "l2": args.l2,
                "lambda_align": args.uniqueness,
                "lambda_div": args.diversity,
            },
        )

        with torch.no_grad():
            X_hat = model.reconstruct()

            test_rmse = completion_rmse(
                X_pred=X_hat,
                test_idx=test_idx,
                test_val=test_val,
            )

        last_rmse = float(test_rmse)

        print(
            f"[Completion] R={R}, "
            f"held-out test RMSE: {last_rmse:.6f}"
        )

    return last_rmse
