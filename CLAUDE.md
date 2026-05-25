# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

A handwritten study copy of [mnielsen/neural-networks-and-deep-learning](https://github.com/mnielsen/neural-networks-and-deep-learning), following the book at http://neuralnetworksanddeeplearning.com/. Python 3, ruff-formatted, with added comments and examples.

## Running the code

Scripts must be run from `src/` because `mnist_loader` hardcodes the data path as `../data/mnist.pkl.gz`:

```bash
cd src
python -c "
import mnist_loader, network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
"
```

For `network2.py` (with regularization and monitoring):

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

## Linting

```bash
ruff format src/
ruff check src/
```

## Architecture

**`src/mnist_loader.py`** — loads `data/mnist.pkl.gz`. `load_data_wrapper()` is the entry point used by network code; it reshapes images to `(784,1)` column vectors and one-hot encodes training labels into `(10,1)` vectors. Validation/test labels remain as plain integers — this asymmetry is intentional for efficiency (cost evaluation vs. accuracy evaluation).

**`src/network.py`** — Chapter 1 baseline. `Network` stores weights as a list of 2D numpy arrays (shape `[neurons_out, neurons_in]`) and biases as column vectors. Uses quadratic cost and vanilla SGD with backprop.

**`src/network2.py`** — Chapter 3 improved version. Key differences from `network.py`:
- Pluggable cost classes (`QuadraticCost`, `CrossEntropyCost`) each implementing `fn(a, y)` and `delta(z, a, y)`. Adding a new cost function requires only implementing these two static methods.
- `default_weight_initializer` scales weights by `1/sqrt(n_in)` to avoid saturated neurons at initialization; `large_weight_initializer` replicates the Chapter 1 approach for comparison.
- `SGD` accepts `lmbda` for L2 regularization (applied in `update_mini_batch` as weight decay) and optional monitoring flags that return per-epoch cost/accuracy lists.
- `save`/`load` serialize the network to JSON; `load` uses `getattr(sys.modules[__name__], ...)` to reconstruct the cost class from its name string.


## Rules
1. Do not add ``Co-author by Claude`` in any commit message.
2. Only commit or push when told to do so.
