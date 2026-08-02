import torch
import os
from sae import SparseAutoEncoder
from torch.utils.data import TensorDataset, DataLoader

device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

sae_train_config = {
    'd_in': 512,
    'expand_factor': 4,
    'l1_coeff': 0.05,
    'batch_size': 1024,
    'lr': 1e-3,
    'epochs': 50,
    'eval_every': 100
}

def load_activations(activations_path):
    data = torch.load(activations_path, map_location=device, weights_only=False) 
    acts = data['activations']  # (n_samples, d_mlp)
    meta = {k: v for k, v in data.items() if k != 'activations'}
    print(f"Loaded activations: {tuple(acts.shape)}")
    print(f"Loaded metadata: {meta}")
    return acts

def get_batch(activations, batch_size):
    # shuffle indices, fresh randomization / each call
    rand_indices = torch.randperm(activations.shape[0])
    for i in range(0, activations.shape[0], batch_size):
        batch_idx = rand_indices[i:i+batch_size]
        yield activations[batch_idx]

def get_l1_coeff(step, total_steps, l1_coeff_final, warm_up_frac):
    """Linearly warms up the L1 coefficient from 0 to l1_coeff_final
    over the first `warmup_frac` fraction of training, then hold it constant.
    """
    warmup_steps = max(1, int(warm_up_frac * total_steps))
    if step >= warmup_steps:
        return l1_coeff_final
    return l1_coeff_final * step / warmup_steps

def train(activations_path, config=sae_train_config, save_path=None):
    if save_path is None:
        raise ValueError("save_path is required")
    d_in = config['d_in']
    d_hidden = config['d_in'] * config['expand_factor']
    epoch = config['epochs']
    batch_size = config['batch_size']

    print(f"Expanding {config['d_in']} -> {d_hidden} features")
    acts = load_activations(activations_path)

    sae = SparseAutoEncoder(d_in, d_hidden=d_hidden).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config['lr'])

    total_steps = epoch * (len(acts) // batch_size)


    # Supply data to SAE.
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
        for i, x in enumerate(get_batch(acts, batch_size)):
            current_l1 = 
            loss, metrics = sae.loss(x, config['l1_coeff'])
            optimizer.zero_grad()   # Reset all gradients.
            loss.backward()         # caculate new gradients.
            optimizer.step()        # update all tensor weights with new gradients.
            sae.normalize_decoder_weights() # address shrink f & grow w_d loophole

            if i % config['eval_every'] == 0:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
                    save_data = {
                        'sae_state_dict': sae.state_dict(),
                        'best_loss': best_loss,
                        'metrics': metrics
                    }
                    torch.save(save_data, save_path)


        

