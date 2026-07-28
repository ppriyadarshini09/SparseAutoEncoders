# SparseAutoEncoders

## Personal Thoughts & Questions
*Below are some of my questions & thoughts while trying to understand SAE. These are the things that I would have raised in the classroom or just silently written down in my notes. Slightly unconventional use of README but works best for me!*

#### Thought#1 - Beauty lies in Simplicity
SAE revolves & evolves around fundational Encoder+Decoder mechanics, summed up with 3 equations:

```python
f(x) = ReLU( (x - b_d) @ W_e + b_e ) # Encoder: dense MLP activations -> sparse features
x_hat = f(x) @ W_d + b_d     # Decoder: sparse features -> reconstructed MLP activations
loss = ||x_hat - x||**2 + lambda * sum(f) # MSE + L1 Sparsity
```

with shape

```python
f(x) =          ReLU(  (x - b_d)      @     W_e (t)       +     b_e )
 ↓                      ↓                    ↓                   ↓
[batch_size, d_hidden] [batch_size, d_mlp]  [d_hidden, d_mlp]   [d_hidden]


x_hat =            f(x)        @          W_d (t)     +     b_d
 ↓                  ↓                      ↓                 ↓
[batch_size, d_mlp][batch_size, d_hidden] [d_mlp, d_hidden] [d_mlp]


loss = ||x_hat - x||**2 + lambda * sum(f) # <- scalar value
            ↓                 ↓
    reconstruction loss     sparsity loss
```

#### Question#1 - How the sparsity loss component helps disentangle features

#### Question#2 - Why subtract `b_d` from `x` before Encoding 

The goal of the decoder is to resconstruct input activations `x` and `b_d` is decoder bias vector. Now for all `x`, learned decoder vector `b_d` is baseline component available in all `x_hat` reconstruction. Subtracting this baseline component helps save encoder some of its feature encoding budget on implicitly re-representing that constant baseline in every active feature, it's an efficiency improvement.

#### Question#4 - What is "shrink f and grow w_d" loophole?


#### Question#3 - Why normalized initialization for W_*


#### Question#4 - Why W_e and W_d weights values are same at initialization?



