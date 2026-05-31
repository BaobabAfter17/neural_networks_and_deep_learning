"""E2E tests for network3.py — verify SGD runs on synthetic data.

Synthetic images: 1-channel 6x6 (flat size 36).
First ConvPool:  filter (2,1,3,3), pool (2,2) -> output (batch, 2, 2, 2), flat=8
Second ConvPool: filter (2,2,1,1), pool (1,1) -> output (batch, 2, 2, 2), flat=8
"""
import torch
import pytest
import network3

FLAT = 36        # 1 * 6 * 6
N_CLASSES = 3
IMAGE_SHAPE = (1, 6, 6)
# 6x6 -> conv(3x3) -> 4x4 -> pool(2x2) -> 2x2, with 2 filters: flat = 2*2*2 = 8
CONV1_OUT_FLAT = 8
MINI_BATCH = 4
N_TRAIN, N_VAL, N_TEST = 20, 8, 8


def _tensors(n):
    x = torch.rand(n, FLAT)
    y = torch.randint(0, N_CLASSES, (n,))
    return x, y


@pytest.fixture(scope="module")
def data():
    return _tensors(N_TRAIN), _tensors(N_VAL), _tensors(N_TEST)


def _sgd(layers, data, lmbda=0.0):
    train, val, test = data
    net = network3.Network(layers, mini_batch_size=MINI_BATCH)
    net.SGD(train, epochs=1, mini_batch_size=MINI_BATCH, eta=0.1,
            validation_data=val, test_data=test, lmbda=lmbda)


def test_fc_softmax(data):
    _sgd([
        network3.FullyConnectedLayer(FLAT, 8),
        network3.SoftmaxLayer(8, N_CLASSES),
    ], data)


def test_conv_softmax(data):
    _sgd([
        network3.ConvPoolLayer(filter_shape=(2, 1, 3, 3), image_shape=IMAGE_SHAPE, poolsize=(2, 2)),
        network3.SoftmaxLayer(CONV1_OUT_FLAT, N_CLASSES),
    ], data)


def test_conv_fc_softmax(data):
    _sgd([
        network3.ConvPoolLayer(filter_shape=(2, 1, 3, 3), image_shape=IMAGE_SHAPE, poolsize=(2, 2)),
        network3.FullyConnectedLayer(CONV1_OUT_FLAT, 8),
        network3.SoftmaxLayer(8, N_CLASSES),
    ], data)


def test_two_conv_fc_softmax(data):
    # Second conv: (2,2,2) image, (2,2,1,1) filter, pool(1,1) -> still (batch,2,2,2)
    _sgd([
        network3.ConvPoolLayer(filter_shape=(2, 1, 3, 3), image_shape=IMAGE_SHAPE, poolsize=(2, 2)),
        network3.ConvPoolLayer(filter_shape=(2, 2, 1, 1), image_shape=(2, 2, 2), poolsize=(1, 1)),
        network3.FullyConnectedLayer(CONV1_OUT_FLAT, 8),
        network3.SoftmaxLayer(8, N_CLASSES),
    ], data)


def test_fc_with_dropout(data):
    _sgd([
        network3.FullyConnectedLayer(FLAT, 8, p_dropout=0.5),
        network3.SoftmaxLayer(8, N_CLASSES),
    ], data)


def test_relu_activation(data):
    _sgd([
        network3.FullyConnectedLayer(FLAT, 8, activation_fn=network3.ReLU),
        network3.SoftmaxLayer(8, N_CLASSES),
    ], data)


def test_sgd_with_lmbda(data):
    _sgd([
        network3.ConvPoolLayer(filter_shape=(2, 1, 3, 3), image_shape=IMAGE_SHAPE, poolsize=(2, 2)),
        network3.FullyConnectedLayer(CONV1_OUT_FLAT, 8),
        network3.SoftmaxLayer(8, N_CLASSES),
    ], data, lmbda=0.1)
