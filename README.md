# SparseAutoEncoders

## Personal Thoughts & Questions
*Below are most ofmy questions & thoughts while trying to understand SAE. These are the things that I would have asked the Professor in the classroom or just silently write down in my notes. Slightly unconventional use of README, please feel free to ignore!*

### Thought#1 - Beauty lies in Simplicity
SAE runs revolves and evolves around on fundational Encoder+Decoder mechanics, summed up with 3 equations:

```python
f(x) = ReLU( (x - b_d) @ W_e + b_e ) # Encoder: dense MLP activations -> sparse features
x_hat = W_d @ f(x) + b_d     # Decoder: sparse features -> reconstructed MLP activations
loss = ||x_hat - x||**2 + lambda * sum(f) # MSE + L1 Sparsity
```

with shape

```python
f(x) =          ReLU(  (x - b_d)      @     W_e (t)       +     b_e )
 ↓                      ↓                    ↓                   ↓
[batch_size, d_hidden] [batch_size, d_mlp]  [d_hidden, d_mlp]   [d_hidden]


x_hat =            f(x)        @          W_d (t)     +     b_d
 ↓                  ↓                      ↓                 ↓
[batch_size, d_mlp][batch_size, d_hidden] [d_mlp, d_hidden] [d_hidden]


loss = ||x_hat - x||**2 + lambda * sum(f) # <- scalar value
```

### Question#1 - Why subtract `b_d` from `x` before Encoding 


### Question#2 - Why normalized initialization for W_*


### Question#3 - Why W_e and W_d weights values are same at initialization?



