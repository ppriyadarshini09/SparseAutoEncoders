"""
Sparse AutoEncoder - decompose dense MLP activations of LLM models (miniGPT here)
into overcomplete, sparse set of features (Bricken et al., "Towards
Monosemanticity", https://transformer-circuits.pub/2023/monosemantic-features).

Core Equations:
    f(x) = ReLU( W_e @ (x - b_d) + b_e ) # [Encoder] dense activations -> sparse features
    x_hat = W_d @ f(x) + b_d # [Decoder] sparse features -> reconstruction
    L = ||x - x_hat||_2^2 + lambda * sum(f) # [Loss] Resconstruction (MSE) + L1 Sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoEncoder(nn.Module):
    """
    A single hidden-layer(which is decomposed/sparsed) sparse autoencoder
    (dictionary learning model).

    Args:
        d_in (int): Dimension of input activations (d_mlp).
        d_hidden (int): Dimension of overcomplete learned features (dictionary size).
                        Typically 4x-8x.
    """

    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden

        # NOTE: torch.empty() doesn't zero initialize, it just allocates
        # chunk of memory with whatever garbage values are already there.
        self.W_e = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_e = nn.Parameter(torch.zeros(d_hidden))

        self.W_d = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_d = nn.Parameter(torch.zeros(d_in))

        # Weights values are drawn from a standard normal distribution
        # of mean 0 and standard deviation 1, in place.
        # If not initialized properly, will lead to instability (huge initial
        # losses, exploding gradients) in training - wasting initial steps.
        nn.init.normal_(self.W_d)
        self.W_d.data = F.normalize(self.W_d.data, dim=0)
        # Transpose and copy as separate instance, not shared as these 2
        # tensors will learn different weights.
        self.W_e.data = self.W_d.data.t().clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the sparse representation of input activations.

        Args:
            x (torch.Tensor): Input activations of shape (batch_size, d_in).

        Returns:
            torch.Tensor: Sparse representation of input activations of shape (batch_size, d_hidden).
        """
        return F.relu((x - self.b_d) @ self.W_e.t() + self.b_e)

    def decode(self, f_x: torch.Tensor) -> torch.Tensor:
        """Reconstructs the input activations from the sparse features

        Args:
            f_x (torch.Tensor): Sparse representation of input activations of shape (batch_size, d_hidden).

        Returns:
            torch.Tensor: Reconstructed input activations of shape (batch_size, d_in).
        """
        return f_x @ self.W_d.t() + self.b_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoder-decoder forward pass

        Args:
            x (torch.Tensor): Input activations of shape (batch_size, d_in).

        Returns:
            torch.Tensor: Reconstructed input activations of shape (batch_size, d_in).
            torch.Tensor: Sparse representation of input activations of shape (batch_size, d_hidden).
        """
        f_x = self.encode(x)
        x_hat = self.decode(f_x)
        return x_hat, f_x

    def loss(self, x: torch.Tensor, l1_coeff: float) -> float:
        """Computes reconstruction loss with L1 regularization to enforce sparsity.

        Args:
            x (torch.Tensor): Input activations of shape (batch_size, d_in).
            l1_coeff (float): Coefficient for L1 regularization.

        Returns:
            float: Total loss value.
        """
        x_hat, f_x = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = f_x.sum(-1).mean()
        total_loss = recon_loss + (l1_coeff * l1_loss)

        with torch.no_grad():
            # L0 norm (number of active neurons per example in batch)
            # dim=-1 - sum across the column in 2d-shape matrix (batch_size, d_hidden)
            l0_norm = (f_x > 0).float().sum(dim=-1).mean().item()
            # Dead features: # of feature didn't fire at all for any example in the batch
            dead_frac = (f_x == 0).float().sum(dim=0).mean().item()

        metrics = {
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "l0": l0_norm,
            "dead_frac": dead_frac,
        }
        return total_loss, metrics
