# SparseAutoEncoders

## How to run?

Create a copy of colab and run all cells in "rain SAE on Weak MiniGPT and Inspect features". 

Example of meaningful feature learned, where this feature fires on capital "W" after a newline, pretty clean and nameable pattern for initial attempt. Retune the miniGPT and SAE parameter to play around more. 

```
==== Feature 9 : density stats ====
Fires on 23516/199936 tokens (0.1176 of all positions)
When active -- mean: 0.400 max: 2.165 min: 0.000

==== Feature 9: 10 activating examples ====
Rank  1, activation=2.165 ...ou have been ere now, and what you are;\n[[W]] ithal, what I have been, and what I am.
Rank  2, activation=2.143 ...,\nAnd presently repair to Crosby Place;\n[[W]] here, after I have solemnly interr'd\nAt
Rank  3, activation=2.136 ...roat to thee and to thy ancient malice;\n[[W]] hich not to cut would show thee but a f
Rank  4, activation=2.134 ...threshold. Why, thou Mars! I tell thee,\n[[W]] e have a power on foot; and I had purpo
Rank  5, activation=2.122 ...ud to do't.\n\nBRUTUS:\nI heard him swear,\n[[W]] ere he to stand for consul, never would
Rank  6, activation=2.116 ...should find you lions, finds you hares;\n[[W]] here foxes, geese: you are no surer, no
Rank  7, activation=2.108 ... dark spirit, in 's nervy arm doth lie;\n[[W]] hich, being advanced, declines, and the
Rank  8, activation=2.103 ...ling them with us, the honour'd number,\n[[W]] ho lack not virtue, no, nor power, but 
Rank  9, activation=2.101 ...w'st thou sugar on that bottled spider,\n[[W]] hose deadly web ensnareth thee about?\nF
Rank 10, activation=2.096 ... so thy breast encloseth my poor heart;\n[[W]] ear both of them, for both of them are 
```

## Personal Thoughts & Questions
*Below are some of my thoughts and questions while trying to understand SAE. These are the things that I would have raised in the classroom or just silently written down in my notes. Slightly unconventional use of README but works best for me!*

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

#### Question#1 - How sparsity loss component helps disentangle features

Recall each neuron in MLP layer is polysemantic that is one neuron can fire for multiple unrelated concepts. LLMs inherently learns to superimpose multiple concepts onto a neuron as there will be more concepts in the training data than number of parameters (/neurons) that model has. The understanding is that the unit of concept are not the neurons, but the direction (a.k.a. feature) vector of these neurons.

In order to distangle meaningful concepts, the idea is to learn expanded representation where each feature represent a unique concept in the training data, which "sparsity loss" component of the loss function helps achieve. The training goal is to minimize overall loss by finding a balance between "reconstruct well" and "use few features" to represent a concepts in training data, that means it will try to keep only a handful of features (out of d_hidden) non-zero. When only a handful of features are active, model can't spread explanations across many overlapping / correlated feature, and each active feature is under pressure eto justify its own ativation cost individually, so the features that are reduant are wasteful, and penalty is paid per active feature. Two feature covering the same concept cost more than one feature covering it.

#### Question#2 - Why subtract `b_d` from `x` before Encoding 

The goal of the decoder is to resconstruct input activations `x` and `b_d` is decoder bias vector. Now for all `x`, learned decoder vector `b_d` is baseline component available in all `x_hat` reconstruction. Subtracting this baseline component helps save encoder some of its feature encoding budget on implicitly re-representing that constant baseline in every active feature, it's an efficiency improvement.

#### Question#3 - What is "shrink f and grow W_d" loophole?

Remember `x_hat = W_d @ f(x) + b_d` meaning there are infinite combinations to achieve same `x_hat`. For instance, dividing `f(x)` by 10 and multiplying `W_d` will produce same `x_hat` and it will ,lower the loss for free since `f(x)` is part of loss computation and `W_d` is not. At the end, all the expanded features will trend towards dead and having learnt nothing except constructing the input.

#### Question#4 - Why normalize decoder weights after each step

To fix above "shrink f and grow W_d" problem. Normalizing `W_d` prevents very high magnitude weights in `W_d`.

#### Question#4 - Why is W_e initialized as the transpose of W_d?

A feature is only useful if what the encoder detects and what the decoder outputs agree -- otherwise firing it adds noise, not signal, to the reconstruction. Random init gives every feature a mismatched detector and output from day one. Tying W_e = W_d.T at init means every feature starts "aligned": firing it immediately helps reconstruction, giving gradient descent a clean signal instead of noise to refine from. They're free to drift apart after that -- only the starting point is shared.



