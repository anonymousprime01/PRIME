'''
Load irregular tensor datasets and create reconstruction or completion splits.

This module loads synthetic or real-world tensors, creates observation masks,
and optionally caches generated data splits.
'''

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from typing import List, Optional, Union
from utils import generate_irregular_tensor


def get_device() -> torch.device:
    '''
    Select an available CUDA device or the CPU.

    Output:
        Selected PyTorch device.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    return device


def get_tensor_path(data_name: str) -> Path:
    '''
    Return the full tensor file path for a real-world dataset.

    Input:
        data_name: Dataset name.

    Output:
        Path of the corresponding tensor file.
    '''
    path_map = {
        "nasdaq": "./data/tensor/nasdaq_tensor.pkl",
        "forbes": "./data/tensor/forbes2000_tensor.pkl",
        "pems": "./data/tensor/pems_tensor.pkl",
        "volume": "./data/tensor/VolumeData_tensor.pkl",
        "electricity": "./data/tensor/electricity_tensor.pkl",
        "metr": "./data/tensor/metr_tensor.pkl",
    }

    if data_name not in path_map:
        raise ValueError(f"Unknown data name: {data_name}")

    return Path(path_map[data_name])


def get_cache_path(
    data_name: str,
    task: str,
    missing_ratio: float,
    seed: int,
) -> Path:
    '''
    Create a cache path for a generated data split.

    Inputs:
        data_name: Dataset name.
        task: Reconstruction or completion.
        missing_ratio: Held-out entry ratio.
        seed: Random seed used to generate the split.

    Output:
        Cache file path.
    '''
    cache_dir = Path("./data/split")
    cache_dir.mkdir(parents=True, exist_ok=True)

    ratio = str(missing_ratio).replace(".", "p")
    seed_tag = "" if seed is None else f"_seed_{seed}"

    return (
        cache_dir
        / (
            f"{data_name}_{task}_missing_ratio_"
            f"{ratio}{seed_tag}.pkl"
        )
    )


def to_tensor_list(X, device: torch.device):
    '''
    Convert tensor slices to float32 PyTorch tensors.

    Inputs:
        X: Collection of tensor slices.
        device: Target PyTorch device.

    Output:
        List of float32 tensors on the target device.
    '''
    tensor_list = []

    for x in X:
        if isinstance(x, torch.Tensor):
            tensor = x.detach().clone().float().to(device)
        else:
            tensor = torch.as_tensor(
                x,
                dtype=torch.float32,
                device=device,
            )

        tensor_list.append(tensor)

    return tensor_list


def load_full_tensor(
    data_name: str,
    seed: int,
    device: torch.device,
):
    '''
    Load a complete synthetic or real-world irregular tensor.

    Inputs:
        data_name: Dataset name.
        seed: Random seed for synthetic data generation.
        device: Target PyTorch device.

    Output:
        List of complete tensor slices.
    '''
    if data_name == "synthetic":
        X = generate_irregular_tensor(seed=seed)
        return to_tensor_list(X, device)

    path = get_tensor_path(data_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Full tensor file does not exist: {path}"
        )

    with path.open("rb") as file:
        data = pickle.load(file)

    if isinstance(data, dict):
        if "X" not in data:
            raise ValueError(
                "Pickle dictionary must contain key 'X'. "
                f"Got keys: {list(data.keys())}"
            )

        X = data["X"]

    else:
        X = data

    return to_tensor_list(X, device)


def _detach_to_cpu(obj: Any) -> Any:
    '''
    Recursively detach tensors and move them to the CPU.
    '''
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()

    if isinstance(obj, list):
        return [_detach_to_cpu(item) for item in obj]

    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(item) for item in obj)

    if isinstance(obj, dict):
        return {
            key: _detach_to_cpu(value)
            for key, value in obj.items()
        }

    return obj


def _move_to_device(obj: Any, device: torch.device) -> Any:
    '''
    Recursively move tensors to the selected device.
    '''
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, list):
        return [
            _move_to_device(item, device)
            for item in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            _move_to_device(item, device)
            for item in obj
        )

    if isinstance(obj, dict):
        return {
            key: _move_to_device(value, device)
            for key, value in obj.items()
        }

    return obj


def save_cache(path: Path, data: dict) -> None:
    '''
    Save a tensor split to a pickle cache file.

    Inputs:
        path: Output cache path.
        data: Data split and metadata.

    Output:
        None. A pickle file is written to disk.
    '''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cpu_data = _detach_to_cpu(data)

    with path.open("wb") as file:
        pickle.dump(cpu_data, file)


def load_cache(path: Path, device: torch.device) -> dict:
    '''
    Load a cached data split and move tensors to a device.

    Inputs:
        path: Cache file path.
        device: Target PyTorch device.

    Output:
        Cached data dictionary.
    '''
    path = Path(path)

    with path.open("rb") as file:
        data = pickle.load(file)

    return _move_to_device(data, device)


def check_same_num_columns(X) -> int:
    '''
    Verify that all slices are matrices with the same column count.

    Input:
        X: List of tensor slices.

    Output:
        Shared number of columns.
    '''
    if not X:
        raise ValueError("The tensor contains no slices.")

    if X[0].ndim != 2:
        raise ValueError(
            f"Each slice must be two-dimensional. Got {X[0].shape}."
        )

    J = X[0].shape[1]

    for k, x in enumerate(X):
        if x.ndim != 2:
            raise ValueError(
                f"X[{k}] must be two-dimensional. Got {x.shape}."
            )

        if x.shape[1] != J:
            raise ValueError(
                "All slices must have the same number of columns. "
                f"X[0].shape[1]={J}, but "
                f"X[{k}].shape[1]={x.shape[1]}."
            )

    return J


def get_length_info(X):
    '''
    Return the row lengths of irregular tensor slices.

    Input:
        X: List of tensor slices.

    Output:
        Slice lengths, maximum length, and minimum length.
    '''
    if not X:
        raise ValueError("The tensor contains no slices.")

    length = [x.shape[0] for x in X]
    max_length = max(length)
    min_length = min(length)

    return length, max_length, min_length


def make_full_mask(X, device: torch.device):
    '''
    Create observation masks containing only True values.

    Inputs:
        X: List of tensor slices.
        device: Target PyTorch device.

    Output:
        List of full observation masks.
    '''
    return [
        torch.ones_like(
            x,
            dtype=torch.bool,
            device=device,
        )
        for x in X
    ]


def make_missing_split(
    X_full,
    missing_ratio: float = 0.3,
    seed: int = 0,
    device: Union[torch.device, str] = "cpu",
):
    '''
    Split tensor entries into training, validation, and test sets.

    The missing ratio represents the total held-out ratio. Half of the
    held-out entries are used for validation and the rest for testing.

    Inputs:
        X_full: Complete tensor slices.
        missing_ratio: Total held-out ratio.
        seed: Random seed for entry sampling.
        device: Target PyTorch device.

    Output:
        Training slices, training masks, validation indices and values,
        and test indices and values.
    '''
    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError(
            "missing_ratio must satisfy 0 <= missing_ratio < 1. "
            f"Got {missing_ratio}."
        )

    if missing_ratio == 0:
        X_train = [
            x.detach().clone().float().to(device)
            for x in X_full
        ]
        train_mask = make_full_mask(X_full, device)

        return X_train, train_mask, None, None, None, None

    rng = np.random.default_rng(seed)

    X_train = []
    train_mask = []
    valid_idx = []
    valid_val = []
    test_idx = []
    test_val = []

    for k, x in enumerate(X_full):
        x = x.detach().clone().float().to(device)

        Ik, J = x.shape
        total = Ik * J

        if total < 3:
            raise ValueError(
                "Each slice must contain at least three entries "
                "to preserve training, validation, and test entries. "
                f"X[{k}] contains {total} entries."
            )

        num_holdout = int(total * missing_ratio)

        # Keep at least one validation entry, one test entry,
        # and one observed training entry.
        num_holdout = max(2, num_holdout)
        num_holdout = min(num_holdout, total - 1)

        num_valid = num_holdout // 2
        num_test = num_holdout - num_valid

        sampled_flat = rng.choice(
            total,
            size=num_holdout,
            replace=False,
        )

        sampled = [
            (int(index // J), int(index % J))
            for index in sampled_flat
        ]

        valid_sampled = sampled[:num_valid]
        test_sampled = sampled[num_valid:]

        mask = torch.ones_like(
            x,
            dtype=torch.bool,
            device=device,
        )
        x_train = x.clone()

        sampled_rows = torch.tensor(
            [i for i, _ in sampled],
            dtype=torch.long,
            device=device,
        )
        sampled_cols = torch.tensor(
            [j for _, j in sampled],
            dtype=torch.long,
            device=device,
        )

        mask[sampled_rows, sampled_cols] = False
        x_train[sampled_rows, sampled_cols] = 0.0

        valid_rows = torch.tensor(
            [i for i, _ in valid_sampled],
            dtype=torch.long,
            device=device,
        )
        valid_cols = torch.tensor(
            [j for _, j in valid_sampled],
            dtype=torch.long,
            device=device,
        )
        test_rows = torch.tensor(
            [i for i, _ in test_sampled],
            dtype=torch.long,
            device=device,
        )
        test_cols = torch.tensor(
            [j for _, j in test_sampled],
            dtype=torch.long,
            device=device,
        )

        X_train.append(x_train)
        train_mask.append(mask)

        valid_idx.append(valid_sampled)
        test_idx.append(test_sampled)

        valid_val.append(
            x[valid_rows, valid_cols].detach().clone()
        )
        test_val.append(
            x[test_rows, test_cols].detach().clone()
        )

    return (
        X_train,
        train_mask,
        valid_idx,
        valid_val,
        test_idx,
        test_val,
    )


def read_data(
    data_name: str,
    seed: int,
    device: torch.device,
    task: str = "reconstruction",
    missing_ratio: float = 0.0,
    use_cache: bool = True,
):
    '''
    Load a tensor and prepare a reconstruction or completion split.

    Inputs:
        data_name: Dataset name.
        seed: Random seed.
        device: Target PyTorch device.
        task: Reconstruction or completion.
        missing_ratio: Total held-out entry ratio.
        use_cache: Whether to load and save split caches.

    Output:
        Training slices, masks, validation data, test data,
        slice lengths, and maximum slice length.
    '''
    if task not in {"reconstruction", "completion"}:
        raise ValueError(f"Unknown task: {task}")

    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError(
            "missing_ratio must satisfy 0 <= missing_ratio < 1. "
            f"Got {missing_ratio}."
        )

    if task == "completion" and missing_ratio <= 0:
        raise ValueError(
            "Completion requires missing_ratio > 0."
        )

    is_real = data_name != "synthetic"

    cache_path = get_cache_path(
        data_name=data_name,
        task=task,
        missing_ratio=missing_ratio,
        seed=seed,
    )

    if is_real and use_cache and cache_path.exists():
        data = load_cache(cache_path, device)

        print("============== loaded cached data ==============")
        print(f"cache_path: {cache_path}")
        print(f"data_name: {data_name}")
        print(f"task: {task}")
        print(f"K: {len(data['X_train'])}")
        print(f"min length: {min(data['length'])}")
        print(f"max length: {data['max_length']}")
        print(f"missing_ratio: {missing_ratio}")

        return (
            data["X_train"],
            data["train_mask"],
            data["valid_idx"],
            data["valid_val"],
            data["test_idx"],
            data["test_val"],
            data["length"],
            data["max_length"],
        )

    X_full = load_full_tensor(
        data_name=data_name,
        seed=seed,
        device=device,
    )

    J = check_same_num_columns(X_full)
    length, max_length, min_length = get_length_info(X_full)

    if task == "reconstruction" and missing_ratio == 0:
        X_train = X_full
        train_mask = make_full_mask(X_full, device)

        valid_idx = None
        valid_val = None
        test_idx = None
        test_val = None

    else:
        (
            X_train,
            train_mask,
            valid_idx,
            valid_val,
            test_idx,
            test_val,
        ) = make_missing_split(
            X_full=X_full,
            missing_ratio=missing_ratio,
            seed=seed,
            device=device,
        )

    data = {
        "X_train": X_train,
        "train_mask": train_mask,
        "valid_idx": valid_idx,
        "valid_val": valid_val,
        "test_idx": test_idx,
        "test_val": test_val,
        "length": length,
        "max_length": max_length,
    }

    if is_real and use_cache:
        save_cache(cache_path, data)

        print("============== saved cached data ==============")
        print(f"cache_path: {cache_path}")

    print("============== complete load data ==============")
    print(f"data_name: {data_name}")
    print(f"task: {task}")
    print(f"K: {len(X_train)}")
    print(f"J: {J}")
    print(f"min length: {min_length}")
    print(f"max length: {max_length}")
    print(f"missing_ratio: {missing_ratio}")

    return (
        X_train,
        train_mask,
        valid_idx,
        valid_val,
        test_idx,
        test_val,
        length,
        max_length,
    )


def read_reconstruction_data(
    data_name: str,
    seed: int,
    device: torch.device,
    missing_ratio: float = 0.0,
    use_cache: bool = True,
):
    '''
    Load data for a tensor reconstruction task.

    Inputs:
        data_name: Dataset name.
        seed: Random seed.
        device: Target PyTorch device.
        missing_ratio: Optional held-out ratio.
        use_cache: Whether to use split caching.

    Output:
        Reconstruction training data and split metadata.
    '''
    return read_data(
        data_name=data_name,
        seed=seed,
        device=device,
        task="reconstruction",
        missing_ratio=missing_ratio,
        use_cache=use_cache,
    )


def read_completion_data(
    data_name: str,
    missing_ratio: float,
    seed: int,
    device: torch.device,
    use_cache: bool = True,
):
    '''
    Load data for a tensor completion task.

    Inputs:
        data_name: Dataset name.
        missing_ratio: Total held-out ratio.
        seed: Random seed.
        device: Target PyTorch device.
        use_cache: Whether to use split caching.

    Output:
        Completion training data and split metadata.
    '''
    return read_data(
        data_name=data_name,
        seed=seed,
        device=device,
        task="completion",
        missing_ratio=missing_ratio,
        use_cache=use_cache,
    )
