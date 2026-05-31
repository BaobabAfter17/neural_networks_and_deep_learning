"""E2E tests for network2.py — verify SGD runs on synthetic data."""
import os
import tempfile

import numpy as np
import network2
from network2 import CrossEntropyCost, QuadraticCost

N_IN, N_HIDDEN, N_OUT = 8, 5, 3
N_TRAIN, N_EVAL = 20, 10
MINI_BATCH = 5


def _training_data(n=N_TRAIN):
    data = []
    for _ in range(n):
        x = np.random.randn(N_IN, 1)
        y = np.zeros((N_OUT, 1))
        y[np.random.randint(N_OUT)] = 1.0
        data.append((x, y))
    return data


def _eval_data(n=N_EVAL):
    return [(np.random.randn(N_IN, 1), int(np.random.randint(N_OUT))) for _ in range(n)]


def test_cross_entropy_cost():
    net = network2.Network([N_IN, N_HIDDEN, N_OUT])
    net.SGD(_training_data(), 1, MINI_BATCH, 0.5)


def test_quadratic_cost():
    net = network2.Network([N_IN, N_HIDDEN, N_OUT], cost=QuadraticCost)
    net.SGD(_training_data(), 1, MINI_BATCH, 0.5)


def test_l2_regularization():
    net = network2.Network([N_IN, N_HIDDEN, N_OUT])
    net.SGD(
        _training_data(), 1, MINI_BATCH, 0.5,
        lmbda=5.0, evaluation_data=_eval_data(),
    )


def test_all_monitoring_flags():
    # vectorized_result in network2 is hardcoded to 10 dims (MNIST convention),
    # so monitor_evaluation_cost requires a 10-output network.
    n_out_10 = 10

    def training_data_10():
        data = []
        for _ in range(N_TRAIN):
            x = np.random.randn(N_IN, 1)
            y = np.zeros((n_out_10, 1))
            y[np.random.randint(n_out_10)] = 1.0
            data.append((x, y))
        return data

    def eval_data_10():
        return [(np.random.randn(N_IN, 1), int(np.random.randint(n_out_10)))
                for _ in range(N_EVAL)]

    net = network2.Network([N_IN, N_HIDDEN, n_out_10])
    eval_cost, eval_acc, train_cost, train_acc = net.SGD(
        training_data_10(), 1, MINI_BATCH, 0.5,
        evaluation_data=eval_data_10(),
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True,
    )
    assert len(eval_cost) == 1
    assert len(eval_acc) == 1
    assert len(train_cost) == 1
    assert len(train_acc) == 1


def test_large_weight_initializer():
    net = network2.Network([N_IN, N_HIDDEN, N_OUT])
    net.large_weight_initializer()
    net.SGD(_training_data(), 1, MINI_BATCH, 0.5)


def test_save_load_and_predict():
    net = network2.Network([N_IN, N_HIDDEN, N_OUT])
    net.SGD(_training_data(), 1, MINI_BATCH, 0.5)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name
    try:
        net.save(fname)
        loaded = network2.load(fname)
        out = loaded.feedforward(np.random.randn(N_IN, 1))
        assert out.shape == (N_OUT, 1)
    finally:
        os.unlink(fname)
