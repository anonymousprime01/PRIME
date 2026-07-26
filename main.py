'''
Main entry point for PARAFAC2-based tensor reconstruction and completion experiments.

This module parses experiment settings, initializes random seeds and devices,
and runs the selected task.
'''

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
import numpy as np
import torch
import warnings

from tasks.tensor_reconstruction import run_reconstruction
from tasks.tensor_completion import run_completion

from typing import Callable, List


def parse_int_list(s: str) -> List[int]:
    '''
    Convert a comma-separated string into a list of integers.
    '''
    s = (s or "").strip()

    if not s:
        return []

    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    '''
    Parse command-line arguments for the experiment.
    '''
    parser = argparse.ArgumentParser()

    # Basic setting
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)

    # Task setting
    parser.add_argument(
        "--task",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "completion"],
        help="reconstruction / completion",
    )

    # Dataset setting
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        help="synthetic / sps500 / korea / nasdaq / nyse / volume / pems / forbes / japan",
    )

    # Completion setting
    parser.add_argument("--missing_ratio", type=float, default=0.2)

    # Model setting
    parser.add_argument(
        "--model",
        type=str,
        default="pmask",
        choices=["parafac2", "pmask", "nomask"],
    )

    parser.add_argument("--R", type=int, default=20)
    parser.add_argument(
        "--R_list",
        type=str,
        default="20",
        help="Comma-separated ranks, e.g., 16,32,64.",
    )
    parser.add_argument("--mask_ratio", type=float, default=0.15)

    # Trainer setting
    parser.add_argument(
        "--trainer",
        type=str,
        default="gd",
        choices=["als", "gd"],
        help="als / gd",
    )

    # GD setting
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving best GD checkpoints. Use an empty string to disable.",
    )

    # Regularization
    parser.add_argument("--smoothness", type=float, default=10)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--uniqueness", type=float, default=1e-3)
    parser.add_argument(
        "--lambda_div",
        dest="diversity",
        type=float,
        default=1,
        help="Diversity regularization weight (lambda_div).",
    )
    parser.add_argument(
        "--skip_artifacts",
        action="store_true",
        help="Do not write checkpoints, curves, plots, SVDs, or reconstructed tensors.",
    )

    return parser.parse_args()


def set_random_seed(seed: int = 0):
    '''
    Set random seeds for reproducible experiments.
    '''
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu):
    '''
    Select a CUDA device when available, otherwise use the CPU.
    '''
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")

    return torch.device("cpu")


def main():
    '''
    Initialize the experiment and run the selected task.
    '''
    args = parse_args()
    set_random_seed(args.seed)

    device = get_device(args.gpu)
    print(f"device using: {device}")

    if args.task == "reconstruction":
        run_reconstruction(args, device)

    elif args.task == "completion":
        run_completion(args, device)


if __name__ == "__main__":
    main()
