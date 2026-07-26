'''
PARAFAC2 models for irregular tensor reconstruction and completion.

This module provides a positive prototype-masked PARAFAC2 model trained by
gradient descent and a standard PARAFAC2 model trained by alternating least
squares.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import khatri_rao


# ---------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------
def masked_sse(model, X, mask, rec_fn=None):
    '''
    Compute the sum of squared reconstruction errors over observed entries.
    '''
    rec_fn = model.reconstruct_slice if rec_fn is None else rec_fn

    sse = torch.zeros((), device=model.device)
    n_obs = 0

    for k in range(model.K):
        mask_k = mask[k].to(model.device, dtype=torch.bool)
        x_true = X[k].to(model.device)
        x_pred = rec_fn(k)

        diff = x_pred[mask_k] - x_true[mask_k]

        sse = sse + torch.sum(diff ** 2)
        n_obs += int(mask_k.sum().item())

    return sse, max(n_obs, 1)


def l2(params):
    '''
    Compute the sum of squared parameter values.
    '''
    loss = 0.0

    for p in params:
        loss = loss + torch.sum(p ** 2)

    return loss


def align(Uk, Phi):
    '''
    Compute the PARAFAC2 Gram-matrix alignment penalty.
    '''
    loss = 0.0

    for U in Uk:
        diff = U.T @ U - Phi
        loss = loss + torch.sum(diff ** 2)

    return loss


@torch.no_grad()
def masked_nre_from_model(model, X, mask, eps=1e-12):
    '''
    Compute relative Frobenius error over observed entries.
    '''
    sse, _ = masked_sse(model, X, mask)

    den = torch.zeros((), device=model.device)

    for k in range(model.K):
        mask_k = mask[k].to(model.device, dtype=torch.bool)
        x_true = X[k].to(model.device)
        den = den + torch.sum(x_true[mask_k] ** 2)

    return torch.sqrt(sse / den.clamp_min(eps))


# ---------------------------------------------------------------------
# Prototype-masked PARAFAC2
# ---------------------------------------------------------------------
class PARAFAC2_PMask(nn.Module):
    '''
    Positive prototype-masked PARAFAC2 model.

    The model constructs row- and column-side masks by combining positive
    prototype dictionaries with positive usage coefficients.

    Inputs:
        length: Row size of each tensor slice.
        J: Shared number of columns.
        R: Base latent rank.
        L: Number of mask prototypes.
        device: PyTorch device.
        init_scale: Parameter initialization scale.
        proto_init: Prototype initialization method.
        eps: Minimum positive mask value.

    Output:
        Reconstructed irregular tensor slices.
    '''

    def __init__(
        self,
        length,
        J,
        R,
        L,
        device,
        init_scale=1.0,
        proto_init="rand",
        eps=1e-6,
    ):
        super().__init__()

        if len(length) == 0:
            raise ValueError("length must contain at least one slice.")

        if J <= 0:
            raise ValueError(f"J must be positive. Got J={J}.")

        if R <= 0:
            raise ValueError(f"R must be positive. Got R={R}.")

        if L <= 0:
            raise ValueError(f"L must be positive. Got L={L}.")

        self.length = list(length)
        self.J = J
        self.R = R
        self.L = L
        self.K = len(length)
        self.device = device
        self.eps = eps

        self.V = nn.Parameter(
            init_scale
            * torch.randn(
                J,
                R,
                dtype=torch.float32,
                device=device,
            )
        )

        self.W = nn.Parameter(
            init_scale
            * torch.randn(
                self.K,
                R,
                dtype=torch.float32,
                device=device,
            )
        )

        self.Uk = nn.ParameterList([
            nn.Parameter(
                init_scale
                * torch.randn(
                    Ik,
                    R,
                    dtype=torch.float32,
                    device=device,
                )
            )
            for Ik in self.length
        ])

        one_raw = math.log(math.exp(1.0) - 1.0)

        def init_mask_raw(shape):
            if proto_init == "one":
                return one_raw + 0.05 * torch.randn(
                    *shape,
                    dtype=torch.float32,
                    device=device,
                )

            if proto_init == "rand":
                return init_scale * torch.randn(
                    *shape,
                    dtype=torch.float32,
                    device=device,
                )

            raise ValueError(f"Unknown proto_init: {proto_init}")

        # Shared prototype dictionaries with shape (L, R).
        self.D_U_raw = nn.Parameter(
            init_mask_raw((L, R))
        )
        self.D_V_raw = nn.Parameter(
            init_mask_raw((L, R))
        )

        # Slice-specific row-side usage matrices with shape (I_k, L).
        self.A_U_raw = nn.ParameterList([
            nn.Parameter(
                init_mask_raw((Ik, L))
            )
            for Ik in self.length
        ])

        # Shared column-side usage matrix with shape (J, L).
        self.A_V_raw = nn.Parameter(
            init_mask_raw((J, L))
        )

        self.register_buffer(
            "Phi",
            torch.eye(
                R,
                dtype=torch.float32,
                device=device,
            ),
        )

        self.update_phi()

        print(
            "==================== complete initializing "
            "PARAFAC2_PMask ===================="
        )

    def positive(self, x):
        '''
        Convert unconstrained mask parameters into positive values.
        '''
        return F.softplus(x) + self.eps

    def U_mask(self, k):
        '''
        Construct row-side masks for tensor slice k.

        Input:
            k: Slice index.

        Output:
            Tensor with shape (I_k, L, R).
        '''
        D_U = self.positive(self.D_U_raw)
        A_U = self.positive(self.A_U_raw[k])

        return A_U[:, :, None] * D_U[None, :, :]

    def V_mask(self, k=None):
        '''
        Construct shared column-side masks.

        Output:
            Tensor with shape (J, L, R).
        '''
        D_V = self.positive(self.D_V_raw)
        A_V = self.positive(self.A_V_raw)

        return A_V[:, :, None] * D_V[None, :, :]

    def rec_slice(self, k):
        '''
        Reconstruct tensor slice k using masked latent components.

        Input:
            k: Slice index.

        Output:
            Reconstructed matrix with shape (I_k, J).
        '''
        U = self.Uk[k]
        V = self.V
        W = self.W[k]

        M_U = self.U_mask(k)
        M_V = self.V_mask()

        I_k = U.shape[0]

        U_cat = (
            U[:, None, :] * M_U
        ).reshape(
            I_k,
            self.L * self.R,
        )

        V_cat = (
            V[:, None, :] * M_V
        ).reshape(
            self.J,
            self.L * self.R,
        )

        W_cat = W.repeat(self.L)

        return ((U_cat * W_cat) @ V_cat.T) / self.L

    def reconstruct_slice(self, k):
        return self.rec_slice(k)

    def reconstruct(self):
        '''
        Reconstruct all tensor slices.

        Output:
            List of reconstructed matrices.
        '''
        return [
            self.rec_slice(k)
            for k in range(self.K)
        ]

    def forward(self):
        return self.reconstruct()

    @torch.no_grad()
    def update_phi(self):
        '''
        Update the shared Gram matrix from slice-specific row factors.
        '''
        gram_sum = torch.zeros(
            self.R,
            self.R,
            dtype=torch.float32,
            device=self.device,
        )

        for U in self.Uk:
            gram_sum = gram_sum + U.T @ U

        self.Phi.copy_(
            gram_sum / max(self.K, 1)
        )

    def smoothness_reg(self):
        '''
        Penalize differences between adjacent rows of the row factors.
        '''
        loss = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=self.device,
        )

        for U in self.Uk:
            if U.shape[0] <= 1:
                continue

            diff = U[1:] - U[:-1]
            loss = loss + diff.pow(2).sum()

        return loss

    def align_reg(self):
        '''
        Compute the PARAFAC2 Gram-matrix alignment penalty.
        '''
        return align(self.Uk, self.Phi)

    def uniqueness_regularization(self):
        return self.align_reg()

    def dictionary_diversity_reg(self, eps=1e-12):
        '''
        Penalize similarity between different prototype dictionaries.
        '''
        if self.L <= 1:
            return torch.zeros(
                (),
                dtype=torch.float32,
                device=self.device,
            )

        D_U = self.positive(self.D_U_raw)
        D_V = self.positive(self.D_V_raw)

        D_U = F.normalize(
            D_U,
            p=2,
            dim=1,
            eps=eps,
        )
        D_V = F.normalize(
            D_V,
            p=2,
            dim=1,
            eps=eps,
        )

        G_U = D_U @ D_U.T
        G_V = D_V @ D_V.T

        off_diag = ~torch.eye(
            self.L,
            dtype=torch.bool,
            device=G_U.device,
        )

        loss_U = G_U[off_diag].pow(2).mean()
        loss_V = G_V[off_diag].pow(2).mean()

        return 0.5 * (loss_U + loss_V)

    def masked_loss(
        self,
        X,
        train_mask,
        lambda_reg=1e-3,
        lambda_align=0,
        lambda_smooth=10,
        lambda_div=1,
    ):
        '''
        Compute reconstruction and regularization losses.

        Inputs:
            X: Target tensor slices.
            train_mask: Observation masks.
            lambda_reg: L2 regularization weight.
            lambda_align: Alignment regularization weight.
            lambda_smooth: Smoothness regularization weight.
            lambda_div: Prototype diversity weight.

        Output:
            Scalar training loss.
        '''
        rec_sse, _ = masked_sse(
            self,
            X,
            train_mask,
            rec_fn=self.rec_slice,
        )

        D_U = self.positive(self.D_U_raw)
        D_V = self.positive(self.D_V_raw)

        A_U = [
            self.positive(A_U_k)
            for A_U_k in self.A_U_raw
        ]
        A_V = self.positive(self.A_V_raw)

        l2_loss = l2([
            self.V,
            self.W,
            D_U,
            D_V,
            A_V,
            *A_U,
        ])

        align_loss = self.align_reg()
        smooth_loss = self.smoothness_reg()
        diversity_loss = self.dictionary_diversity_reg()

        return (
            rec_sse
            + lambda_reg * l2_loss
            + lambda_align * align_loss
            + lambda_smooth * smooth_loss
            + lambda_div * diversity_loss
        )

    @torch.no_grad()
    def masked_nre(self, X, mask):
        '''
        Compute relative Frobenius error over observed entries.
        '''
        return masked_nre_from_model(
            self,
            X,
            mask,
        )


# ---------------------------------------------------------------------
# PARAFAC2 ALS
# ---------------------------------------------------------------------
class PARAFAC2_ALS(nn.Module):
    '''
    Standard PARAFAC2 model trained by alternating least squares.

    The slice-specific row factor is represented as U_k = Q_k H, where Q_k
    has orthonormal columns and H is shared across tensor slices.

    Inputs:
        length: Row size of each tensor slice.
        J: Shared number of columns.
        R: Target rank.
        device: PyTorch device.
        init_scale: Parameter initialization scale.

    Output:
        Reconstructed irregular tensor slices.
    '''

    def __init__(
        self,
        length,
        J,
        R,
        device,
        init_scale=1.0,
    ):
        super().__init__()

        if len(length) == 0:
            raise ValueError("length must contain at least one slice.")

        if J <= 0:
            raise ValueError(f"J must be positive. Got J={J}.")

        if R <= 0:
            raise ValueError(f"R must be positive. Got R={R}.")

        if min(length) < R:
            raise ValueError(
                "PARAFAC2 requires every slice length I_k >= R. "
                f"Got min(length)={min(length)}, R={R}."
            )

        self.length = list(length)
        self.J = J
        self.R = R
        self.K = len(length)
        self.device = device

        self.H = nn.Parameter(
            init_scale
            * torch.randn(
                R,
                R,
                dtype=torch.float32,
                device=device,
            )
        )

        self.V = nn.Parameter(
            init_scale
            * torch.rand(
                J,
                R,
                dtype=torch.float32,
                device=device,
            )
        )

        self.W = nn.Parameter(
            init_scale
            * torch.rand(
                self.K,
                R,
                dtype=torch.float32,
                device=device,
            )
        )

        self.Qk = nn.ParameterList([
            nn.Parameter(
                torch.zeros(
                    Ik,
                    R,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            for Ik in self.length
        ])

        self.register_buffer(
            "Y",
            torch.zeros(
                R,
                J,
                self.K,
                dtype=torch.float32,
                device=device,
            ),
        )

        print(
            "==================== complete initializing "
            "PARAFAC2_ALS ====================="
        )

    def U_k(self, k):
        '''
        Return the slice-specific row factor U_k = Q_k H.
        '''
        return self.Qk[k] @ self.H

    def reconstruct_slice(self, k):
        '''
        Reconstruct tensor slice k.

        Input:
            k: Slice index.

        Output:
            Reconstructed matrix with shape (I_k, J).
        '''
        return (
            (self.Qk[k] @ self.H * self.W[k])
            @ self.V.T
        )

    def rec_slice(self, k):
        return self.reconstruct_slice(k)

    def reconstruct(self):
        '''
        Reconstruct all tensor slices.

        Output:
            List of reconstructed matrices.
        '''
        return [
            self.reconstruct_slice(k)
            for k in range(self.K)
        ]

    def forward(self):
        return self.reconstruct()

    def make_initial_working_tensor(
        self,
        X,
        train_mask,
    ):
        '''
        Initialize missing entries with zeros for ALS training.

        Inputs:
            X: Input tensor slices.
            train_mask: Observation masks.

        Output:
            List of initialized working tensor slices.
        '''
        X_work = []

        for k in range(self.K):
            x_true = X[k].to(self.device)
            mask_k = train_mask[k].to(
                self.device,
                dtype=torch.bool,
            )

            xk = torch.where(
                mask_k,
                x_true,
                torch.zeros_like(x_true),
            )

            X_work.append(xk)

        return X_work

    def make_working_tensor(
        self,
        X,
        train_mask,
    ):
        '''
        Replace missing entries with current model predictions.

        Inputs:
            X: Input tensor slices.
            train_mask: Observation masks.

        Output:
            List of updated working tensor slices.
        '''
        X_hat = self.reconstruct()
        X_work = []

        for k in range(self.K):
            x_true = X[k].to(self.device)
            mask_k = train_mask[k].to(
                self.device,
                dtype=torch.bool,
            )

            xk = torch.where(
                mask_k,
                x_true,
                X_hat[k].detach(),
            )

            X_work.append(xk)

        return X_work

    @torch.no_grad()
    def als_update_Q(self, X_work):
        '''
        Update the slice-specific orthonormal matrices Q_k.

        Input:
            X_work: Current working tensor slices.

        Output:
            None. Q_k and projected tensors are updated in place.
        '''
        for k in range(self.K):
            X_k = X_work[k].to(self.device)

            S_k = torch.diag(self.W[k])

            target = (
                X_k
                @ self.V
                @ S_k
                @ self.H.T
            )

            Z_k, _, Pk_T = torch.linalg.svd(
                target,
                full_matrices=False,
            )

            Q_new = (
                Z_k[:, :self.R]
                @ Pk_T[:self.R, :]
            )

            self.Qk[k].copy_(Q_new)

        for k in range(self.K):
            self.Y[:, :, k].copy_(
                self.Qk[k].T
                @ X_work[k].to(self.device)
            )

    @torch.no_grad()
    def als_update_HVW(self):
        '''
        Update H, V, and W using alternating least squares.
        '''
        Y1 = (
            self.Y
            .permute(0, 2, 1)
            .reshape(
                self.R,
                self.K * self.J,
            )
        )

        KR_WV = khatri_rao(
            self.W,
            self.V,
        )

        H_gram = (
            (self.W.T @ self.W)
            * (self.V.T @ self.V)
        )

        self.H.copy_(
            Y1
            @ KR_WV
            @ torch.linalg.pinv(H_gram)
        )

        Y2 = (
            self.Y
            .permute(1, 2, 0)
            .reshape(
                self.J,
                self.K * self.R,
            )
        )

        KR_WH = khatri_rao(
            self.W,
            self.H,
        )

        V_gram = (
            (self.W.T @ self.W)
            * (self.H.T @ self.H)
        )

        self.V.copy_(
            Y2
            @ KR_WH
            @ torch.linalg.pinv(V_gram)
        )

        Y3 = (
            self.Y
            .permute(2, 1, 0)
            .reshape(
                self.K,
                self.J * self.R,
            )
        )

        KR_VH = khatri_rao(
            self.V,
            self.H,
        )

        W_gram = (
            (self.V.T @ self.V)
            * (self.H.T @ self.H)
        )

        self.W.copy_(
            Y3
            @ KR_VH
            @ torch.linalg.pinv(W_gram)
        )

    def masked_loss(
        self,
        X,
        train_mask,
        lambda_l2=0.0,
    ):
        '''
        Compute the observed reconstruction loss and L2 penalty.

        Inputs:
            X: Target tensor slices.
            train_mask: Observation masks.
            lambda_l2: L2 regularization weight.

        Output:
            Scalar training loss.
        '''
        rec_sse, n_obs = masked_sse(
            self,
            X,
            train_mask,
        )

        return (
            rec_sse / n_obs
            + lambda_l2
            * l2([
                self.H,
                self.V,
                self.W,
            ])
            / n_obs
        )

    @torch.no_grad()
    def masked_nre(self, X, mask):
        '''
        Compute relative Frobenius error over observed entries.
        '''
        return masked_nre_from_model(
            self,
            X,
            mask,
        )

