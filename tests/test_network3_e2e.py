"""E2E tests for network3.py.

Part 1: API consistency — verify the public interface matches the original spec.
         No data required; runs instantly.

Part 2: Convergence — load full MNIST via load_data_shared, train 3 epoch with
         the book's parameters (mini_batch_size=10, eta=0.1), assert test
         accuracy >= 95%.  A fixed random seed makes every run deterministic:
         once the threshold passes it will always pass for a correct
         implementation, and a regression (wrong gradient, wrong cost, etc.)
         will pull accuracy well below 95%.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import network3

DATA_PATH = Path(__file__).parent.parent / "data" / "mnist.pkl.gz"
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accuracy(net, x, y, batch_size=1000):
    """Evaluate accuracy over a full dataset in chunks."""
    n = x.shape[0] // batch_size
    return float(
        np.mean(
            [
                net._accuracy(
                    x[j * batch_size : (j + 1) * batch_size],
                    y[j * batch_size : (j + 1) * batch_size],
                )
                for j in range(n)
            ]
        )
    )


# ---------------------------------------------------------------------------
# Part 1 — API consistency
# ---------------------------------------------------------------------------


def test_activation_functions():
    z = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.equal(network3.linear(z), z)
    assert (network3.ReLU(z) >= 0).all()
    s = network3.sigmoid(z)
    assert ((s > 0) & (s < 1)).all()
    t = network3.tanh(z)
    assert (t.abs() < 1).all()


def test_load_data_shared_signature():
    sig = inspect.signature(network3.load_data_shared)
    assert "filename" in sig.parameters


def test_conv_pool_layer_api():
    sig = inspect.signature(network3.ConvPoolLayer.__init__)
    params = sig.parameters
    assert {"filter_shape", "image_shape", "poolsize", "activation_fn"}.issubset(params)
    assert params["poolsize"].default == (2, 2)
    assert params["activation_fn"].default is network3.sigmoid

    layer = network3.ConvPoolLayer(filter_shape=(2, 1, 3, 3), image_shape=(1, 6, 6))
    assert layer.w.shape == torch.Size([2, 1, 3, 3])
    assert layer.b.shape == torch.Size([2])
    assert len(layer.params) == 2


def test_fully_connected_layer_api():
    sig = inspect.signature(network3.FullyConnectedLayer.__init__)
    params = sig.parameters
    assert {"n_in", "n_out", "activation_fn", "p_dropout"}.issubset(params)
    assert params["p_dropout"].default == 0.0
    assert params["activation_fn"].default is network3.sigmoid

    layer = network3.FullyConnectedLayer(n_in=10, n_out=5)
    assert layer.w.shape == torch.Size([10, 5])
    assert layer.b.shape == torch.Size([5])
    assert len(layer.params) == 2


def test_softmax_layer_api():
    sig = inspect.signature(network3.SoftmaxLayer.__init__)
    params = sig.parameters
    assert {"n_in", "n_out", "p_dropout"}.issubset(params)
    assert params["p_dropout"].default == 0.0

    layer = network3.SoftmaxLayer(n_in=10, n_out=5)
    assert layer.w.shape == torch.Size([10, 5])
    assert layer.b.shape == torch.Size([5])
    assert len(layer.params) == 2
    assert callable(layer.cost_from_output)


def test_network_api():
    sig = inspect.signature(network3.Network.__init__)
    params = sig.parameters
    assert {"layers", "mini_batch_size", "use_gpu"}.issubset(params)
    assert params["use_gpu"].default is False

    layers = [network3.FullyConnectedLayer(4, 3), network3.SoftmaxLayer(3, 2)]
    net = network3.Network(layers, mini_batch_size=2)
    assert net.layers is layers
    assert len(net.params) == 4  # w + b for each of the 2 layers


def test_sgd_signature():
    sig = inspect.signature(network3.Network.SGD)
    assert list(sig.parameters.keys()) == [
        "self",
        "training_data",
        "epochs",
        "mini_batch_size",
        "eta",
        "validation_data",
        "test_data",
        "lmbda",
    ]
    assert sig.parameters["lmbda"].default == 0.0


# ---------------------------------------------------------------------------
# Part 2 — Convergence  (requires MNIST data)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mnist():
    if not DATA_PATH.exists():
        pytest.skip("MNIST data not found at " + str(DATA_PATH))
    return network3.load_data_shared(str(DATA_PATH))


def test_conv_fc_softmax_converges(mnist):
    """Canonical architecture from README: Conv(20) + FC(100) + Softmax(10).

    3 epochs with the book's parameters should reach ~96-97% on MNIST.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    training_data, validation_data, test_data = mnist
    net = network3.Network(
        [
            network3.ConvPoolLayer(filter_shape=(20, 1, 5, 5), image_shape=(1, 28, 28)),
            network3.FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
            network3.SoftmaxLayer(n_in=100, n_out=10),
        ],
        mini_batch_size=10,
    )
    net.SGD(
        training_data,
        epochs=3,
        mini_batch_size=10,
        eta=0.1,
        validation_data=validation_data,
        test_data=test_data,
    )
    assert _accuracy(net, *test_data) >= 0.95


def test_conv_fc_softmax_with_lmbda_converges(mnist):
    """Same architecture with L2 regularization (lmbda=0.1)."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    training_data, validation_data, test_data = mnist
    net = network3.Network(
        [
            network3.ConvPoolLayer(filter_shape=(20, 1, 5, 5), image_shape=(1, 28, 28)),
            network3.FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
            network3.SoftmaxLayer(n_in=100, n_out=10),
        ],
        mini_batch_size=10,
    )
    net.SGD(
        training_data,
        epochs=3,
        mini_batch_size=10,
        eta=0.1,
        validation_data=validation_data,
        test_data=test_data,
        lmbda=0.1,
    )
    assert _accuracy(net, *test_data) >= 0.95
