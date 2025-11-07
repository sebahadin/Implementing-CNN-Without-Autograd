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

All images are normalized using the standard MNIST mean (( \mu = 0.1307 )) and standard deviation (( \sigma = 0.3081 )).

---

## Model Architecture

Implemented in the `build_cnn()` function as a custom `Sequential` container:

```
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

* Operation:
  [
  Y[n, c_{out}, h, w] = \sum_{c=0}^{C_{in}-1} \sum_{i=0}^{k_H-1} \sum_{j=0}^{k_W-1} X[n, c, h+i, w+j] \cdot W[c_{out}, c, i, j] + b[c_{out}]
  ]
* Implemented with `F.unfold` to extract image patches, and `torch.bmm` for batch matrix multiplication.

### ReLU (see `ReLU.forward()`)

* Elementwise nonlinearity:
  [
  \text{ReLU}(x) = \max(0, x)
  ]
* Implemented with `torch.clamp(x, min=0)`

### MaxPool2D (see `MaxPool2D.forward()`)

* Pooling operation over ( k \times k ) windows:
  [
  Y[n, c, h, w] = \max_{(i,j) \in k \times k} X[n, c, h+i, w+j]
  ]
* Implemented with `F.unfold`, `view`, and `max(dim=2)`

### Flatten (see `Flatten.forward()`)

* Reshapes input:
  [
  X \in \mathbb{R}^{N \times C \times H \times W} \Rightarrow X' \in \mathbb{R}^{N \times (C \cdot H \cdot W)}
  ]
* Done with `x.view(x.size(0), -1)`

### Linear (see `Linear.forward()`)

* Fully connected layer:
  [
  Y = XW + b
  ]
* Implemented as `x @ self.w + self.b`

### CrossEntropy (see `CrossEntropy.forward()`)

* Given logits ( z \in \mathbb{R}^{N \times C} ), targets ( y \in {0,...,C-1} ):
  [
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log \left( \frac{\exp(z_{i, y_i})}{\sum_j \exp(z_{i,j})} \right)
  ]

---

## Manual Gradient Computation (Backward Pass)

Each layer implements `bwd(out, *args)` called via `backward()`, updating `x.g` (i.e., gradients).

### ReLU (see `ReLU.bwd()`)

* Derivative:
  [
  \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \mathbb{1}_{x > 0}
  ]
* Code:

```python
x.g = out.g * (x > 0).float()
```

### Linear (see `Linear.bwd()`)

* Derivatives:
  [
  \frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}, \quad
  \frac{\partial L}{\partial b} = \sum \frac{\partial L}{\partial Y}, \quad
  \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T
  ]
* Code:

```python
self.w.g = x.T @ out.g
self.b.g = out.g.sum(0)
x.g = out.g @ self.w.T
```

### Conv2D (see `Conv2D.bwd()`)

* Gradient of filter:
  [
  \frac{\partial L}{\partial W} = \sum_{n=1}^N G[n] \cdot X_{\text{unfold}}[n]^T
  ]
* Gradient of input (reconstructed):
  [
  \frac{\partial L}{\partial X} = F.\text{fold}(W^T \cdot G)
  ]
* Code uses `F.unfold`, `F.fold`, and `torch.bmm`

### MaxPool2D (see `MaxPool2D.bwd()`)

* Only propagate gradient to the max index in each patch:
  [
  \frac{\partial L}{\partial x_{i}} =
  \begin{cases}
  \frac{\partial L}{\partial y} & \text{if } x_{i} = \max(x) \
  0 & \text{otherwise}
  \end{cases}
  ]
* Code:

```python
Xg_cols.scatter_(2, self.max_idx.unsqueeze(2), G_cols.unsqueeze(2))
```

### CrossEntropy (see `CrossEntropy.bwd()`)

* Softmax:
  [
  \hat{y}_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
  ]
* Gradient:
  [
  \frac{\partial L}{\partial z} = \frac{1}{N} (\hat{y} - y)
  ]
* Code:

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
     [
     \theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}
     ]
     via `sgd_step(params, lr)`

---

## Utility Functions

* `params_of(net)` collects all trainable parameters (W, b)
* `sgd_step(params, lr)` updates parameters:
  [
  \theta := \theta - \eta \cdot \theta.g
  ]
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

Every `Module` has `.forward()` and `.bwd()` functions to compute gradients, replacing PyTorch's automatic differentiation.

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
