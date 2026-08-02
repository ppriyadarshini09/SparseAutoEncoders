"""
Stage 4 - Interpret a trained SparseAutoencoder.

What this module does:
1. Loads saved trained SAE and the saved activations.
2. Runs every activation through the SAE'e encoder to get the sparse feature matrix f, shape (N, d_hidden)
3. For one chosen feature index, find the top-K token positions where it fires the strongest,
   and prints a text window around each one.
4. Prints density stats (how often this feature fire at all, and its activation distribution)
   To understand if a feature is rare-but-specific or degenerate (dead on everything).
"""

import os
import torch

from sae import SparseAutoEncoder
from tokenizer import CharTokenizer

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def load_sae(saved_path):
    ckpt = torch.load(saved_path, map_location=device, weights_only=False)
    model = SparseAutoEncoder(
        d_in = ckpt['d_in'],
        d_hidden = ckpt['d_hidden']
    ).to(device)

    model.load_state_dict(ckpt['sae_state_dict'])
    model.eval()
    print(f"Loaded SAE: d_in={ckpt['d_in']}, d_hidden={ckpt['d_hidden']}")
    return model


def rebuild_tokenizer(data_path: str):
    text = open(data_path, "r").read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    return tok, data


@torch.no_grad()
def compute_feature_matrix(
    sae: SparseAutoEncoder, 
    activations: torch.Tensor, 
    batch_size: int = 4096):
    """
    Computes expanded / sparse feature representation (d_hidden) for
    all tokens (in batch_size).
    """
    all_f = []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size].to(device)
        f = sae.encode(batch)
        all_f.append(f.cpu())
    # feature for all token activations.
    return torch.cat(all_f, dim=0)


def show_top_activating_examples(
        feature_idx: int,
        f: torch.Tensor,
        positions: torch.Tensor,
        data: torch.Tensor,
        tok: CharTokenizer,
        top_k: int = 10,
        context_chars: int = 40,
):
    """
    Print the top-K text windows where the feature fires strongest.

    Args:
        feature_idx (int): Index of the feature to inspect (index of one out of d_hidden)
        f (torch.Tensor): Sparse feature matrix of shape (N, d_hidden); N = # of tokens
        positions (torch.Tensor): Positions of the tokens in the data
        data (torch.Tensor): tokenized data
        tok (CharTokenizer): Tokenizer for decoding tokens
        top_k (int): Number of top activating examples to print
        context_chars (int): Number of characters to show around each activating example
    """
    
    feature_acts = f[:, feature_idx] # Get activations for all tokens for the specified feature
    top_values, top_indices = torch.topk(feature_acts, k=top_k)

    print(f"\n==== Feature {feature_idx}: {top_k} activating examples ====")

    for rank, (value, idx) in enumerate(zip(top_values.tolist(), top_indices.tolist())):
        pos = positions[idx].item()
        start = max(0, pos - context_chars)
        end = min(len(data), pos + context_chars)

        context_ids = data[start:end].tolist()
        context_text = tok.decode()

        marker_offset = pos - start
        highlighted = (context_text[:marker_offset]
                       + "[[" 
                       + context_text[marker_offset:marker_offset+1] 
                       + "]] "
                       + context_text[marker_offset+1:])
        highlighted = highlighted.replace("\n", "\\n")
        print(f"Rank {rank+1:2d}, activation={value:.3f} ...{highlighted}")
                                        

def show_feature_density(feature_idx: int, f: torch.Tensor):
    """
    Show the density/distribution stats for a single feature, essentially
    how often & strong it fires.

    Args:
        feature_idx (int): Index of the feature to inspect
        f (torch.Tensor): Sparse feature matrix of shape (N, d_hidden)
    """
    feature_acts = f[:, feature_idx]
    n_total = len(feature_acts)
    n_active = (feature_acts > 0).sum().item()
    density = n_active / n_total

    print(f"\n==== Feature {feature_idx} : density stats ====")
    print(f"Fires on {n_active}/{n_total} tokens ({density:.4f} of all positions)")
    if n_active > 0:
        active_vals = feature_acts[feature_acts > 0]
        print(f"When active -- mean: {active_vals.mean():.3f} "
              f"max: {active_vals.max():.3f} min: {active_vals.min():.3f}")
    else:
        print(f"DEAD FEATURE -- never activates on this dataset.")


def summarize_feature_stats(f: torch.Tensor):
    """Prints corpus wide summary: how many features are dead, how many 
    are extremely dense (likely uninterpretable), and the overall L0.

    Args:
        f (torch.Tensor): Sparse feature matrix of shape (N, d_hidden)
    """

    n_total, d_hidden = f.shape
    density_per_feature = (f > 0).float().mean(dim=0) # (d_hidden,)
    dead = (density_per_feature == 0).sum().item()
    very_dense = (density_per_feature > 0.5).sum().item()
    avg_l0 = (f > 0).float().sum(dim=-1).mean().item() # (N,)

    print(f"\n==== Corpus wide feature stats ====")
    print(f"Total features: {d_hidden}")
    print(f"Dead features: {dead}")
    print(f"Very dense features (fires on >50% of tokens, likely uninterpretable): "
          f"{very_dense} ({very_dense/d_hidden:.1%})")
    print(f"Average L0 (features active per token): {avg_l0:.3f} / {d_hidden}")

def inspect_feature(data_path, saved_activations_path, saved_sae_path):
    tok, data = rebuild_tokenizer(data_path)
    sae = load_sae(saved_sae_path)

    saved_acts = torch.load(saved_activations_path, map_location=device, weights_only=False)
    activations = saved_acts['activations']
    positions = saved_acts['positions']

    f = compute_feature_matrix(sae, activations)
    summarize_feature_stats(f)

