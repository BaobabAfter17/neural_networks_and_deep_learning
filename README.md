# Neural Networks and Deep Learning — Study Implementation

A hands-on reimplementation of [Michael Nielsen's *Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com/), built from scratch in Python 3 as I work through the book. The goal is to deeply understand the mechanics — backpropagation, SGD, regularization, convolutional networks — by writing and annotating every piece of the code myself.

Based on [mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning), with these changes:

- **Python 3** throughout (the original targets Python 2)
- **PyTorch** replaces Theano in `network3.py` (Chapter 6) — Theano is no longer maintained; the port preserves the original public API while using modern tensor operations and autograd
- **ruff** formatting and linting
- Added comments and inline examples while reading

## What's implemented

| File | Chapter | What it covers |
|---|---|---|
| `src/network.py` | 1 | Baseline feedforward network — backprop, quadratic cost, vanilla SGD |
| `src/network2.py` | 3 | Cross-entropy cost, L2 regularization, better weight init, monitoring |
| `src/network3.py` | 6 | Convolutional networks, max pooling, dropout, softmax — **PyTorch** |

All three networks train on MNIST handwritten digits.

## Running

Scripts must be run from `src/` (the data loader hardcodes a relative path to `../data/mnist.pkl.gz`).

**Chapter 1 — basic network:**
```bash
cd src
python -c "
import mnist_loader, network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
"
```

**Chapter 3 — improved network with regularization:**
```bash
cd src
python -c "
import mnist_loader, network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)
"
```

**Chapter 6 — convolutional network (PyTorch):**
```bash
cd src
python -c "
import network3
training_data, validation_data, test_data = network3.load_data_shared()
net = network3.Network([
    network3.ConvPoolLayer(image_shape=(1,28,28), filter_shape=(20,1,5,5), poolsize=(2,2)),
    network3.FullyConnectedLayer(n_in=20*12*12, n_out=100),
    network3.SoftmaxLayer(n_in=100, n_out=10),
], mini_batch_size=10)
net.SGD(training_data, 60, 10, 0.1, validation_data=validation_data, test_data=test_data)
"
```

## Setup

```bash
pip install -r requirements.txt
```

## Tests

`tests/test_network3_e2e.py` verifies that modifications to `network3.py` preserve the original behavior across two dimensions:

**Part 1 — API consistency** (7 tests, instant): checks that every public symbol — activation functions, layer constructors, `Network.__init__`, `Network.SGD` — has the correct signature and attributes.

**Part 2 — Convergence** (2 tests, ~7 s): loads full MNIST via `load_data_shared`, trains the canonical `[Conv(20) → FC(100) → Softmax(10)]` architecture for 1 epoch using the book's parameters (`mini_batch_size=10`, `eta=0.1`), and asserts test accuracy ≥ 90%. A fixed random seed makes results fully deterministic.

```bash
python3 -m pytest tests/
```

## Linting

```bash
ruff format src/
ruff check src/
```
