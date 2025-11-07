# CNN from Scratch: MNIST Classification

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch **without autograd**. It manually defines the forward and backward passes for each layer, revealing the internal mechanisms of automatic differentiation systems.

---

## Overview

* **Objective**: Classify digits from the MNIST dataset.
* **Framework**: PyTorch (only for tensor manipulation and utilities).
* **Key Concepts**:

  * Manual forward propagation
  * Manual backward propagation (gradient derivation)
  * Custom layers (Conv2D, ReLU, MaxPool2D, Flatten, Linear)
  * Manual cross-entropy loss and SGD optimizer
  * Full training loop and performance visualization

---

## Dataset Preparation

MNIST dataset is loaded and normalized with:

* Mean: `0.1307`
* Standard Deviation: `0.3081`

The dataset is split into training (80%), validation (20%), and test sets. Each image is transformed into a tensor and normalized.

---

## Model Architecture

The CNN model follows this sequence:

```text
Input (1×28×28)
↓ Conv2D(1→8, 3×3, padding=1)
↓ ReLU
↓ MaxPool2D(2×2)
↓ Conv2D(8→16, 3×3, padding=1)
↓ ReLU
↓ MaxPool2D(2×2)
↓ Flatten
↓ Linear(784 → 10)
Output: Class scores (logits)
```

---

## Forward Pass Details

### Convolution Layer

Given input ( x \in \mathbb{R}^{N \times C_{in} \times H \times W} ) and kernels ( W \in \mathbb{R}^{C_{out} \times C_{in} \times k_H \times k_W} ):

```
y[n, c_out, h, w] = Σ_{c, i, j} x[n, c, h+i, w+j] * W[c_out, c, i, j] + b[c_out]
```

Implemented using:

* `F.unfold` to extract image patches
* `torch.bmm` for batch matrix multiplication

### ReLU

Elementwise activation:

```
ReLU(x) = max(0, x)
```

### Max Pooling

Pools over non-overlapping ( k \times k ) windows:

```
y[n, c, h, w] = max(x[n, c, h:h+k, w:w+k])
```

### Flatten

Reshapes tensor:

```
[N, C, H, W] → [N, C × H × W]
```

### Linear Layer

Fully connected transformation:

```
y = xW + b
```

### Cross-Entropy Loss

For logits ( z \in \mathbb{R}^{N \times C} ) and true class indices ( y \in {0, ..., C-1} ):

```
L = -1/N × Σ_i log(softmax(z_i)[y_i])
```

Implemented with a numerically stable log-softmax computation.

---

## Manual Backpropagation: Gradient Derivations

Every tensor manually stores its gradient in `.g`. Each layer implements a `.bwd()` method to compute gradients.

### ReLU Backpropagation

The derivative of ReLU is:

```
dL/dx = dL/dout * 1(x > 0)
```

Code:

```python
x.g = out.g * (x > 0).float()
```

### Linear Layer Backpropagation

Given ( y = xW + b ):

* Gradient w.r.t. weights:

```
dL/dW = xᵀ @ dL/dy
```

* Gradient w.r.t. bias:

```
dL/db = sum(dL/dy, dim=0)
```

* Gradient w.r.t. input:

```
dL/dx = dL/dy @ Wᵀ
```

Code:

```python
self.w.g = x.T @ out.g
self.b.g = out.g.sum(0)
x.g = out.g @ self.w.T
```

### Conv2D Backpropagation

* Input unfolded: ( X_{unf} )
* Output gradients: ( G )

**Weight gradients** (accumulated across batches):

```
dL/dW = Σ_n G[n] @ X_unf[n]ᵀ
```

**Bias gradient**:

```
dL/db = G.sum(dim=(0, 2))
```

**Input gradient** (folded back into image shape):

```
dL/dx = fold(Wᵀ @ G)
```

### MaxPool2D Backpropagation

Max pooling passes gradients only to max positions from forward pass.

```
dL/dx[i] = dL/dy if x[i] was max else 0
```

Code:

```python
Xg_cols.scatter_(2, max_idx.unsqueeze(2), G_cols.unsqueeze(2))
```

### CrossEntropy Backpropagation

Let ( ,
abla_z L ) be the gradient of the loss w.r.t. the logits:

* Compute softmax:

```
softmax(z) = exp(z) / sum(exp(z))
```

* Gradient of the loss:

```
dL/dz = (softmax - one_hot(y)) / N
```

Code:

```python
logits.g = (softmax - one_hot) / N
```

---

## Training Procedure

1. Shuffle and batch training data
2. For each batch:

   * Forward pass through the network
   * Compute loss using CrossEntropy
   * Manually zero gradients
   * Call `.bwd()` on loss, then `.backward()` on network
   * Update weights using SGD:

     ```text
     param = param - lr * param.g
     ```

---

## Metrics & Evaluation

Training/validation loss and accuracy are tracked and plotted over epochs.
The final test accuracy is computed using the trained model.

---

## Summary: What You Learn About Autograd

This implementation emulates PyTorch's autograd system:

* Tracks all intermediate operations
* Stores tensors for backward pass
* Computes gradients using the chain rule

Every operation's backward computation is written explicitly — giving insight into the core of deep learning frameworks.

---
