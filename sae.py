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
        # If not initialized properly, will lead to instability in training,
        # If not initialized properly, will lead to instability (huge initial
        # losses, exploding gradients) in training - wasting initial training steps.
        nn.init.normal_(self.W_d)
        self.W_d.data = F.normalize(self.W_d.data, dim=0)





