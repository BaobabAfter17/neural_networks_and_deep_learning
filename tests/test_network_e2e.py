"""E2E tests for network.py — verify SGD runs on synthetic data."""
import numpy as np
import network

N_IN, N_HIDDEN, N_OUT = 8, 5, 3
N_TRAIN, N_TEST = 20, 10
MINI_BATCH = 5


def _training_data(n=N_TRAIN):
    data = []
    for _ in range(n):
        x = np.random.randn(N_IN, 1)
        y = np.zeros((N_OUT, 1))
        y[np.random.randint(N_OUT)] = 1.0
        data.append((x, y))
    return data


def _test_data(n=N_TEST):
    return [(np.random.randn(N_IN, 1), int(np.random.randint(N_OUT))) for _ in range(n)]


def test_sgd_without_test_data():
    net = network.Network([N_IN, N_HIDDEN, N_OUT])
    net.SGD(_training_data(), epochs=1, mini_batch_size=MINI_BATCH, eta=0.5)


def test_sgd_with_test_data():
    net = network.Network([N_IN, N_HIDDEN, N_OUT])
    net.SGD(
        _training_data(), epochs=1, mini_batch_size=MINI_BATCH, eta=0.5,
        test_data=_test_data(),
    )
