'''
Training utilities for gradient descent and ALS-based PARAFAC2 models.

This module selects the appropriate trainer, optimizes model parameters,
tracks validation performance, restores the best model, and optionally saves
checkpoints and training curves.
'''

import copy
import csv
import os
import time

import torch

from utils import completion_rmse


def get_trainer(args):
    '''
    Create a trainer from command-line arguments.

    Input:
        args: Parsed experiment arguments.

    Output:
        GDTrainer or ALSTrainer instance.
    '''
    if args.trainer == "als":
        return ALSTrainer(
            max_iter=args.epochs,
            tol=0,
            patience=args.patience,
            log_every=1,
        )

    if args.trainer == "gd":
        return GDTrainer(
            max_iter=args.epochs,
            lr=args.lr,
            lambda_align=args.uniqueness,
            lambda_smooth=args.smoothness,
            lambda_div=args.diversity,
            patience=args.patience,
            l2=args.l2,
            log_every=1,
        )

    raise ValueError(f"Unknown trainer: {args.trainer}")


class GDTrainer:
    '''
    Train a PARAFAC2 model using gradient descent.

    The trainer uses Adam with cosine learning-rate decay. Reconstruction tasks
    select the model with the best training NRE, while completion tasks select
    the model with the best validation RMSE.

    Inputs:
        max_iter: Maximum number of optimization iterations.
        lr: Initial learning rate.
        lambda_align: Alignment regularization weight.
        lambda_smooth: Smoothness regularization weight.
        lambda_div: Diversity regularization weight.
        l2: L2 regularization weight.
        tol: Reconstruction convergence tolerance.
        patience: Validation early-stopping patience.
        min_delta: Minimum validation improvement.
        log_every: Logging interval.
    '''

    def __init__(
        self,
        max_iter=10000,
        lr=1e-3,
        lambda_align=0.1,
        lambda_smooth=10.0,
        lambda_div=100.0,
        l2=1e-4,
        tol=1e-5,
        patience=20,
        min_delta=1e-4,
        log_every=10,
    ):
        self.max_iter = max_iter
        self.lr = lr
        self.lambda_align = lambda_align
        self.lambda_smooth = lambda_smooth
        self.lambda_div = lambda_div
        self.l2 = l2
        self.tol = tol
        self.patience = patience
        self.min_delta = min_delta
        self.log_every = log_every

    def fit(
        self,
        model,
        X,
        train_mask,
        valid_mask=None,
        save_path=None,
        save_meta=None,
        curve_path=None,
    ):
        '''
        Train a model and restore its best state.

        Inputs:
            model: Model implementing masked_loss, masked_nre, and reconstruct.
            X: Training tensor slices.
            train_mask: Observation masks.
            valid_mask: Validation indices and values for completion.
            save_path: Optional checkpoint path.
            save_meta: Optional experiment metadata.
            curve_path: Optional CSV path for the training curve.

        Output:
            Trained model restored to its best recorded state.
        '''
        n_params = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(f"Trainable parameters: {n_params:,}")
        print(
            "Regularization weights | "
            f"lambda_align={self.lambda_align:.6g}, "
            f"lambda_smooth={self.lambda_smooth:.6g}, "
            f"lambda_div={self.lambda_div:.6g}"
        )

        if save_meta is None:
            save_meta = {}

        history = []

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_iter,
            eta_min=self.lr * 0.01,
        )

        is_completion = valid_mask is not None
        metric_name = "RMSE" if is_completion else "NRE"

        best_metric = float("inf")
        best_iter = -1
        best_state = None

        prev_train_loss = None
        patience_count = 0

        t_start = time.time()
        t_last = t_start

        for it in range(self.max_iter):
            optimizer.zero_grad(set_to_none=True)

            train_loss = model.masked_loss(
                X=X,
                train_mask=train_mask,
                lambda_reg=self.l2,
                lambda_align=self.lambda_align,
                lambda_smooth=self.lambda_smooth,
                lambda_div=self.lambda_div,
            )

            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            if hasattr(model, "update_phi"):
                model.update_phi()

            with torch.no_grad():
                if is_completion:
                    valid_idx, valid_val = valid_mask
                    X_hat = model.reconstruct()

                    train_metric = model.masked_nre(
                        X=X,
                        mask=train_mask,
                    )

                    valid_metric = completion_rmse(
                        X_pred=X_hat,
                        test_idx=valid_idx,
                        test_val=valid_val,
                    )

                    current_metric = float(
                        valid_metric.detach().item()
                    )

                    if current_metric < best_metric - self.min_delta:
                        best_metric = current_metric
                        best_iter = it
                        best_state = copy.deepcopy(
                            model.state_dict()
                        )
                        patience_count = 0
                    else:
                        patience_count += 1

                else:
                    train_metric = model.masked_nre(
                        X=X,
                        mask=train_mask,
                    )
                    valid_metric = None

                    current_metric = float(
                        train_metric.detach().item()
                    )

                    if current_metric < best_metric:
                        best_metric = current_metric
                        best_iter = it
                        best_state = copy.deepcopy(
                            model.state_dict()
                        )

                    if prev_train_loss is None:
                        diff = torch.tensor(
                            0.0,
                            device=train_loss.device,
                        )
                        rel_improve = torch.tensor(
                            1.0,
                            device=train_loss.device,
                        )
                    else:
                        diff = (
                            prev_train_loss
                            - train_loss.detach()
                        )
                        rel_improve = (
                            diff
                            / prev_train_loss.abs().clamp_min(1e-12)
                        )

                    prev_train_loss = (
                        train_loss.detach().clone()
                    )

            scheduler.step()

            elapsed_time = time.time() - t_start
            avg_update_time_so_far = (
                elapsed_time / max(it + 1, 1)
            )

            current_lr = optimizer.param_groups[0]["lr"]
            train_nre_value = float(
                train_metric.detach().item()
            )

            history.append(
                {
                    "method": save_meta.get(
                        "method",
                        model.__class__.__name__,
                    ),
                    "dataset": save_meta.get("dataset", ""),
                    "split": save_meta.get("split", ""),
                    "seed": save_meta.get("seed", ""),
                    "epoch": save_meta.get(
                        "epoch",
                        save_meta.get("seed", ""),
                    ),
                    "iter": it,
                    "elapsed_time": float(elapsed_time),
                    "avg_update_time": float(
                        avg_update_time_so_far
                    ),
                    "fit": train_nre_value,
                    "train_nre": train_nre_value,
                    "train_loss": float(
                        train_loss.detach().item()
                    ),
                    "valid_metric": (
                        float(valid_metric.detach().item())
                        if valid_metric is not None
                        else ""
                    ),
                    "best_metric": float(best_metric),
                    "lr": float(current_lr),
                    "n_params": int(n_params),
                }
            )

            should_log = (
                self.log_every is not None
                and self.log_every > 0
                and (
                    it % self.log_every == 0
                    or it == self.max_iter - 1
                )
            )

            if should_log:
                dt = time.time() - t_last
                current_lr = optimizer.param_groups[0]["lr"]

                if is_completion:
                    print(
                        f"Iteration: {it:>3d}/{self.max_iter}, "
                        f"train loss: {train_loss.item():.3e}, "
                        f"train NRE: {train_metric.item():.4f}, "
                        f"valid {metric_name}: {valid_metric.item():.4f}, "
                        f"best valid {metric_name}: "
                        f"{best_metric:.4f}@{best_iter}, "
                        f"patience: {patience_count}/{self.patience}, "
                        f"lr={current_lr:.2e}, "
                        f"step_time={dt:.3f}s"
                    )
                else:
                    print(
                        f"Iteration: {it:>3d}/{self.max_iter}, "
                        f"train loss: {train_loss.item():.3e}, "
                        f"train {metric_name}: "
                        f"{train_metric.item():.4f}, "
                        f"best train {metric_name}: "
                        f"{best_metric:.4f}@{best_iter}, "
                        f"diff: {diff.item():.3e}, "
                        f"rel_improve: {rel_improve.item():.3e}, "
                        f"tol: {self.tol:.1e}, "
                        f"lr={current_lr:.2e}, "
                        f"step_time={dt:.3f}s"
                    )

                t_last = time.time()

            if is_completion and patience_count >= self.patience:
                print(
                    f"Early stopped at iteration {it}. "
                    f"Best validation {metric_name}: "
                    f"{best_metric:.4f}@{best_iter}."
                )
                break

        total_wall_time = time.time() - t_start
        num_updates = len(history)
        avg_update_time = (
            total_wall_time / max(num_updates, 1)
        )

        print(
            f"[Runtime] total wall-clock time: "
            f"{total_wall_time:.3f}s, "
            f"avg/update: {avg_update_time:.4f}s, "
            f"updates: {num_updates}"
        )

        if best_state is not None:
            model.load_state_dict(best_state)

            if hasattr(model, "update_phi"):
                model.update_phi()

            print(
                f"Loaded best model from iteration {best_iter} "
                f"with {metric_name}={best_metric:.4f}."
            )

            if save_path is not None:
                save_dir = os.path.dirname(save_path)

                if save_dir != "":
                    os.makedirs(
                        save_dir,
                        exist_ok=True,
                    )

                torch.save(
                    {
                        "model_state_dict": best_state,
                        "best_iter": best_iter,
                        "best_metric": best_metric,
                        "metric_name": metric_name,
                        "is_completion": is_completion,
                        "meta": save_meta,
                    },
                    save_path,
                )

                print(
                    f"Saved best model checkpoint to: "
                    f"{save_path}"
                )

        if curve_path is not None:
            curve_dir = os.path.dirname(curve_path)

            if curve_dir != "":
                os.makedirs(
                    curve_dir,
                    exist_ok=True,
                )

            fieldnames = [
                "method",
                "dataset",
                "split",
                "seed",
                "epoch",
                "iter",
                "elapsed_time",
                "avg_update_time",
                "fit",
                "train_nre",
                "train_loss",
                "valid_metric",
                "best_metric",
                "lr",
                "n_params",
            ]

            with open(
                curve_path,
                "w",
                newline="",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                writer.writeheader()
                writer.writerows(history)

            print(
                f"Saved training curve to: {curve_path}"
            )

        return model


class ALSTrainer:
    '''
    Train a PARAFAC2 model using alternating least squares.

    The trainer directly updates model factors without backpropagation.
    Missing entries remain zero-filled during ALS updates. Reconstruction tasks
    stop by relative loss improvement, while completion tasks use validation
    RMSE and patience-based early stopping.

    Inputs:
        max_iter: Maximum number of ALS iterations.
        tol: Relative loss-improvement tolerance.
        patience: Validation early-stopping patience.
        min_delta: Minimum validation improvement.
        log_every: Logging interval.
    '''

    def __init__(
        self,
        max_iter=100,
        tol=1e-5,
        patience=10,
        min_delta=1e-4,
        log_every=1,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.min_delta = min_delta
        self.log_every = log_every

    def fit(
        self,
        model,
        X,
        train_mask,
        valid_mask=None,
        save_path=None,
        save_meta=None,
        curve_path=None,
    ):
        '''
        Train an ALS-compatible PARAFAC2 model.

        Inputs:
            model: Model implementing ALS update and evaluation methods.
            X: Zero-filled training tensor slices.
            train_mask: Observation masks used for evaluation.
            valid_mask: Validation indices and values for completion.
            save_path: Reserved checkpoint path.
            save_meta: Reserved experiment metadata.
            curve_path: Reserved training-curve path.

        Output:
            Trained model restored to its best validation state when available.
        '''
        print(
            "==================== train model by ALS "
            "======================"
        )

        n_params = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(f"Model parameters: {n_params:,}")

        is_completion = valid_mask is not None
        metric_name = "RMSE" if is_completion else "NRE"

        prev_train_loss = None

        best_valid_metric = None
        best_iter = None
        best_snap = None
        patience_count = 0

        t_last = time.time()

        for it in range(self.max_iter):
            X_work = X

            model.als_update_Q(X_work)
            model.als_update_HVW()

            loss = model.masked_loss(
                X,
                train_mask,
            )
            train_metric = model.masked_nre(
                X,
                train_mask,
            )

            with torch.no_grad():
                if is_completion:
                    valid_idx, valid_val = valid_mask
                    X_hat = model.reconstruct()

                    valid_metric = completion_rmse(
                        X_pred=X_hat,
                        test_idx=valid_idx,
                        test_val=valid_val,
                    )

                    if best_valid_metric is None:
                        improved = True
                    else:
                        improved = (
                            valid_metric
                            < best_valid_metric - self.min_delta
                        )

                    if improved:
                        best_valid_metric = (
                            valid_metric.detach().clone()
                        )
                        best_iter = it
                        patience_count = 0
                        best_snap = {
                            name: tensor.detach().clone()
                            for name, tensor
                            in model.state_dict().items()
                        }
                    else:
                        patience_count += 1

                    diff = None
                    rel_improve = None

                else:
                    valid_metric = None

                    if prev_train_loss is None:
                        diff = torch.tensor(
                            0.0,
                            device=loss.device,
                        )
                        rel_improve = torch.tensor(
                            1.0,
                            device=loss.device,
                        )
                    else:
                        diff = (
                            prev_train_loss
                            - loss.detach()
                        )
                        rel_improve = (
                            diff
                            / prev_train_loss.abs().clamp_min(1e-12)
                        )

            should_log = (
                self.log_every is not None
                and self.log_every > 0
                and (
                    it % self.log_every == 0
                    or it == self.max_iter - 1
                )
            )

            if should_log:
                dt = time.time() - t_last

                if is_completion:
                    print(
                        f"Iteration: {it:>3d}, "
                        f"train loss: {loss.item():.3e}, "
                        f"train NRE: {train_metric.item():.4f}, "
                        f"valid {metric_name}: "
                        f"{valid_metric.item():.4f}, "
                        f"best valid {metric_name}: "
                        f"{best_valid_metric.item():.4f}, "
                        f"best iter: {best_iter}, "
                        f"patience: "
                        f"{patience_count}/{self.patience}, "
                        f"step_time={dt:.3f}s"
                    )
                else:
                    print(
                        f"Iteration: {it:>3d}, "
                        f"train loss: {loss.item():.3e}, "
                        f"train {metric_name}: "
                        f"{train_metric.item():.4f}, "
                        f"diff: {diff.item():.3e}, "
                        f"rel_improve: "
                        f"{rel_improve.item():.3e}, "
                        f"tol: {self.tol:.1e}, "
                        f"step_time={dt:.3f}s"
                    )

                t_last = time.time()

            if is_completion:
                if patience_count >= self.patience:
                    print(
                        f"Early stopped at iteration {it}. "
                        f"Best validation {metric_name}: "
                        f"{best_valid_metric.item():.4f} "
                        f"at iteration {best_iter}."
                    )
                    break

            else:
                if (
                    prev_train_loss is not None
                    and rel_improve.item() < self.tol
                ):
                    print(
                        f"Converged at iteration {it}."
                    )
                    break

                prev_train_loss = loss.detach().clone()

        if is_completion and best_snap is not None:
            model.load_state_dict(best_snap)

            print(
                f"Restored best validation model from "
                f"iteration {best_iter} with valid "
                f"{metric_name}: "
                f"{best_valid_metric.item():.4f}."
            )

        return model
