"""
Collect MLP activations from trained miniGPT, for SAE training.

This module does the following step-by-step:
1.  Loads trained checkpoints(weights + config), so make sure to save it.
2.  Registers a forward hook on the FFN's GeLU ouput for the chosen block/layer
   ->   this is the "512 neuron MLP layer" after attention step
        (https://github.com/ppriyadarshini09/miniGPT#feed-forward-network-ffn)
3.  Runs eval-mode (no dropouts), no-grad forward passes over the training text
    and collect activations.
"""

import os
import argparse
import torch

from model import GPT
from tokenizer import CharTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_checkpoint(ckpt_path):
    """
    Loads model and config from saved checkpoint path.

    Reconstructs the GPT architecture, loads the trained weights,
    sets the model to eval() mode so that dropout is disable for
    downstream activation collection.

    Args:
        ckpt_path (str): Path to saved trained GPT model

    Returns:
        tuple:
            model (GPT): reconstrcuted model with trained weights in eval mode.
            config (dict): training config saved alongside trained GPT model.
    """
    # checkpoint should have everything you saved while training the model.
    # looks for something:
    # -------------------
    # torch.save({
    #            'step'      : step,
    #            'model'     : model.state_dict(),
    #            'optimizer' : optimizer.state_dict(),
    #            'config'    : config,
    #            'val_loss'  : best_val_loss,
    # }, ckpt_path)
    # --------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embed=config["n_embed"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    # Should be strictly in eval mode, otherwise it will drop weights randomly
    # during forward pass, which willmuddy the entire SAE training.
    model.eval()
    print(f"Loaded checkpoint (step {ckpt['step']}, val_loss {ckpt['val_loss']:4f})")
    print(
        f"Config: n_layers={config['n_layers']} n_embed={config['n_embed']} "
        f"block_size={config['block_size']} -> d_mlp={4 * config['n_embed']}"
    )
    return model, config


def load_full_token_stream(data_path):
    """
    Loads raw text data after tokenization into token stream.

    Args:
        data_path (str): full path to raw text data

    Returns:
        tuple:
            data (tensor): tokenzied data
            tok (CharTokenzier): Tokenzier used
    """
    text = open(data_path, "r").read()
    # TODO: this might fail as data is in miniGPT repository.
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"token stream shape: {tuple(data.shape)}")
    return data, tok


def register_activation_hook(model, layer_idx=-1):
    """Hook the GELU output inside FFN of the chosen block.
    FFN.net is Sequential (Linear, GeLU, Linear, Droput) -> index 1 is GeLU.

    Args:
        model (GPT): instance of reconstructed GPT model from saved checkpoint.
        layer_idx (int): index of tranformer layer block to collect activations.

    Returns:
        tuple:
            handle (RemovableHandle): required to remove the hook
            captured (dict): current activations
    """
    captured = {}

    def hook_fn(module, inp, out):
        captured["activations"] = out.detach()

    block = model.transformer["blocks"][layer_idx]
    handle = block.ffn.net[1].register_forward_hook(hook_fn)
    return handle, captured


@torch.no_grad()
def collect_activations(
    model, data, config, layer_idx=-1, max_tokens=None, batch_size=256
):
    """
    Collect activations by running through data token streams for SAE training.

    Args:
        model (GPT): reconstructed GPT from saved chcekpoint
        data (tensor): tokenzied raw text data
        config (dict): Training config params saved along with model chcekpoints.
        layer_idx (int): tranformer layer to collect activations from
        max_tokens (int): maximum tokens to run through model to collect activations.
        batch_size (int): batch size to group collected activations

    Returns:
        activations (list): list of tensor of size (batch_size, block_size, d_mlp)
        positions (list): list of each batch start positions from data tensor
    """
    block_size = config["block_size"]
    n_positions = len(data) if max_tokens is None else min(max_tokens, len(data))
    n_chunks = n_positions // block_size
    n_positions = n_chunks * block_size  # trim to a whole number of chunks

    handle, captured = register_activation_hook(model, layer_idx)

    all_activations = []
    all_positions = []
    chunk_starts = list(range(0, n_positions, block_size))

    for batch_start in range(0, len(chunk_starts), batch_size):
        batch_chunk_starts = chunk_starts[batch_start : batch_start + batch_size]
        x = torch.stack(data[s : s + block_size] for s in batch_chunk_starts).to(device)

        _ = model(x)  # forward only -> hook fills captured['activations']
        acts = captured["activations"]  # shape: (batch_size, block_size, d_mlp)

        B, T, D = acts.shape
        all_activations.append(acts.reshape(B * T, D).cpu())

        positions = torch.tensor([s + t for s in batch_chunk_starts for t in range(T)])
        all_positions.append(positions)

        done = batch_start + len(batch_chunk_starts)
        if (batch_start // batch_size) % 10 == 0:
            print(f"    processed {done}/{len(chunk_starts)} chunks")

    handle.remove()

    activations = torch.cat(all_activations, dim=0)  # (N, d_mlp)
    positions = torch.cat(all_positions, dim=0)  # (N,)
    return activations, positions


def save_activations(
    ckpt_path, data_path, layer_idx, max_tokens=None, batch_size=256, out_path=None
):
    """
    Saves collected activations to given path.
    Args:
        ckpt_path (str): full path to trained model checkpoints (saved as .pt file)
        data_path (str): full path to raw text file to build the token stream to collect activations.
        layer_idx (int): tranformer layer to collect activations from
        max_tokens (int): maximum tokens to run through model to collect activations.
        batch_size (int): batch size to group collected activations
        out_path (str): full path to save collected activations.

    Raises:
        FileNotFoundError: If ckpt_path or data_path don't exist.
    """
    model, config = load_checkpoint(ckpt_path)
    data, tok = load_full_token_stream(data_path)
    print(f"Total token available: {len(data):,}")

    activations, positions = collect_activations(
        model, data, layer_idx, max_tokens, batch_size
    )

    print(
        f"\nCollected activations: {tuple(activations.shape)}  (N token-positions x d_mlp)"
    )
    print(
        f"Mean: {activations.mean().item():.4f}  Std: {activations.std().item():.4f}  "
        f"Min: {activations.min().item():.4f}  Max: {activations.max().item():.4f}"
    )
    print(
        "Note: these raw GELU activations are NOT sparse (GELU rarely outputs exactly 0) "
        "-> that's expected. The SAE's job later is to impose sparsity on top of this dense signal."
    )

    torch.save(
        {
            "activations": activations,
            "positions": positions,
            "layer_idx": layer_idx,
            "config": config,
            "ckpt_path": ckpt_path,
        },
        out_path,
    )
    print(f"Saved activations to {out_path}")
