import torch
import os
from sae import SparseAutoEncoder

device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

sae_train_config = {
    'd_mlp': 512,
    'expand_factor': 4,
    'l1_coeff': 0.05,
    'batch_size': 1024,
    'lr': 1e-3,
    'epochs': 100,
    'eval_every': 100
}

def load_activations(activations_path):
    data = torch.load(activations_path, map_location=device, weights_only=False) 
    acts = data['activations']  # (n_samples, d_mlp)
    return acts

def get_batch(activations, batch_size):
    for i in range(0, activations.shape[0], batch_size):
        yield activations[i:i+batch_size]

def train(activations_path, config=sae_train_config, save_path=None):
    if save_path is None:
        raise ValueError("save_path is required")

    acts = load_activations(activations_path)
    print(f"Training on activations of shape: {acts.shape}")
    print(f"Expanding {config['d_mlp']} -> {config['d_mlp'] * config['expand_factor']} features")
    sae = SparseAutoEncoder(
        d_in=config['d_mlp'],
        d_hidden=config['d_mlp'] * config['expand_factor']
    ).to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=config['lr'])

    # Supply data to SAE.
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}")
        for i, x in enumerate(get_batch(acts, config['batch_size'])):
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


        

