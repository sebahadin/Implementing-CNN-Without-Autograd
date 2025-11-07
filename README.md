# CNN from Scratch on MNIST

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch **without using autograd**. All layers, forward passes, and backward passes (gradient computations) are written manually to mimic what PyTorch's autograd engine does under the hood.

---

## Project Overview

* **Goal**: Classify handwritten digits from the MNIST dataset.
* **Framework**: PyTorch (for tensors and helper functions only).
* **Main Features**:

  * Manual forward and backward implementations
  * Convolutional layers, ReLU, max pooling, flattening, and fully connected layers
  * Custom cross-entropy loss
  * SGD optimizer (manual)
  * Visualization of training and test performance

---

## Dataset

We use the MNIST handwritten digit dataset:

* **Training set**: 48,000 samples
* **Validation set**: 12,000 samples
* **Test set**: 10,000 samples

All images are normalized using the standard MNIST mean (`μ = 0.1307`) and standard deviation (`σ = 0.3081`).

---

## Model Architecture

Implemented in the `build_cnn()` function as a custom `Sequential` container:

```python
Conv2D(1, 8, kernel_size=3, stride=1, padding=1)  # 28x28 -> 28x28
ReLU()
MaxPool2D(kernel_size=2, stride=2)               # 28x28 -> 14x14
Conv2D(8, 16, kernel_size=3, stride=1, padding=1) # 14x14 -> 14x14
ReLU()
MaxPool2D(kernel_size=2, stride=2)               # 14x14 -> 7x7
Flatten()                                         # 16x7x7 -> 784
Linear(784, 10)                                   # 784 -> 10 (class logits)
```

---

## Forward Pass

Each `Module` subclass implements:

* `__call__(x)`: stores intermediate values, calls `forward()`
* `forward(x)`: computes the output

### Conv2D (see `Conv2D.forward()`)

**Operation:**

```
Y[n, c_out, h, w] = sum_c sum_i sum_j X[n, c, h+i, w+j] * W[c_out, c, i, j] + b[c_out]
```

Implemented using `F.unfold` to extract image patches, and `torch.bmm` for batch matrix multiplication.

### ReLU (see `ReLU.forward()`)

```
ReLU(x) = max(0, x)
```

Implemented with `torch.clamp(x, min=0)`.

### MaxPool2D (see `MaxPool2D.forward()`)

```
Y[n, c, h, w] = max_{(i,j) in window} X[n, c, sh + i, sw + j]
```

Implemented using `F.unfold`, reshaping and `max(dim=2)`.

### Flatten (see `Flatten.forward()`)

```
[N, C, H, W] -> [N, C * H * W]
```

### Linear (see `Linear.forward()`)

```
Y = X @ W + b
```

### CrossEntropy (see `CrossEntropy.forward()`)

Given logits `z ∈ ℝ^{N×C}` and labels `y ∈ {0,...,C-1}`:

```
L = -1/N * sum_i log(exp(z[i, y_i]) / sum_j exp(z[i, j]))
```

---

## Manual Gradient Computation (Backward Pass)

Each layer implements `bwd(out, *args)` called via `backward()`, updating `x.g` (gradients).

### ReLU (see `ReLU.bwd()`)

```
dL/dx = dL/dy * (x > 0)
```

Code:

```python
x.g = out.g * (x > 0).float()
```

### Linear (see `Linear.bwd()`)

```
dL/dW = x.T @ dL/dy
dL/db = sum(dL/dy)
dL/dx = dL/dy @ W.T
```

Code:

```python
self.w.g = x.T @ out.g
self.b.g = out.g.sum(0)
x.g = out.g @ self.w.T
```

### Conv2D (see `Conv2D.bwd()`)

Gradient of weights:

```
dL/dW = sum_n (G[n] @ X_unfold[n].T)
```

Gradient of input:

```
dL/dX = fold(W.T @ G)
```

Implemented using `F.fold`, `F.unfold`, and `torch.bmm`.

### MaxPool2D (see `MaxPool2D.bwd()`)

```
dL/dx[i] = dL/dy if x[i] was max in window, else 0
```

Code:

```python
Xg_cols.scatter_(2, self.max_idx.unsqueeze(2), G_cols.unsqueeze(2))
```

### CrossEntropy (see `CrossEntropy.bwd()`)

```
softmax = exp(logits) / sum(exp(logits))
dL/dlogits = (softmax - one_hot) / N
```

Code:

```python
logits.g = (softmax - one_hot) / N
```

---

## Training Loop

In `train_model()`:

1. Shuffle training set
2. For each batch:

   * Forward pass: `logits = net(X_batch)`

   * Compute loss: `loss = criterion(logits, y_batch)`

   * Zero gradients: `p.g = torch.zeros_like(p)`

   * Backward pass:

     ```python
     criterion.bwd(loss, logits, y_batch)
     net.backward(logits)
     ```

   * SGD update:

     ```
     p = p - lr * p.g
     ```

---

## Utility Functions

* `params_of(net)` collects all trainable parameters (`W`, `b`)
* `sgd_step(params, lr)` updates:

```
p = p - lr * p.g
```

* `accuracy(logits, y)` returns correct classification rate

---

## Results

Training history is visualized:

* Loss vs. Epoch (Train and Validation)
* Accuracy vs. Epoch (Train and Validation)

Final test accuracy is computed and printed.

---

## Manual Backprop and Autograd

This project mirrors PyTorch's autograd engine:

* Stores inputs and outputs of each layer
* Implements backward gradient propagation using chain rule
* Mimics computation graph and parameter updates

Every `Module` has `.forward()` and `.bwd()` methods to compute gradients, replacing PyTorch's automatic differentiation.

---

## File Structure

```bash
.
├── computer_vision_cnn_hw1.py  # Core implementation
└── README.md                   # This file
```

---

## Acknowledgements

* MNIST dataset provided by torchvision

---
