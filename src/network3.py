"""network3.py
~~~~~~~~~~~~~~

A PyTorch-based program for training and running simple neural networks.

Supports several layer types (fully connected, convolutional, max pooling,
softmax), and activation functions (sigmoid, tanh, and rectified linear
units, with more easily added).

Originally based on Michael Nielsen's Theano implementation from
http://neuralnetworksanddeeplearning.com/ (Chapter 6). Rewritten to use
PyTorch, which supports Python 3 and modern NumPy. The public API — layer
classes, Network, load_data_shared, and SGD — is identical to the original.

"""

#### Libraries
import gzip
import pickle

import numpy as np
import torch
import torch.nn.functional as F


# Activation functions for neurons (same names as the original Theano version)
def linear(z):
    return z


def ReLU(z):
    return torch.relu(z)


sigmoid = torch.sigmoid
tanh = torch.tanh


#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    with gzip.open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

    def to_tensors(data):
        x = torch.tensor(data[0], dtype=torch.float32)
        y = torch.tensor(data[1], dtype=torch.long)
        return x, y

    return [
        to_tensors(training_data),
        to_tensors(validation_data),
        to_tensors(test_data),
    ]


#### Main class used to construct and train networks
class Network:
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [p for layer in layers for p in layer.params]

    def _forward(self, x, dropout=False):
        for layer in self.layers:
            x = layer.forward(x, dropout=dropout)
        return x

    def _accuracy(self, x, y):
        with torch.no_grad():
            out = self._forward(x, dropout=False)
            return (torch.argmax(out, dim=1) == y).float().mean().item()

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        validation_data,
        test_data,
        lmbda=0.0,
    ):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        num_training_batches = training_x.shape[0] // mini_batch_size
        num_validation_batches = validation_x.shape[0] // mini_batch_size
        num_test_batches = test_x.shape[0] // mini_batch_size

        best_validation_accuracy = 0.0
        test_accuracy = 0.0
        for epoch in range(epochs):
            for mb_idx in range(num_training_batches):
                iteration = num_training_batches * epoch + mb_idx
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))

                xb = training_x[
                    mb_idx * mini_batch_size : (mb_idx + 1) * mini_batch_size
                ]
                yb = training_y[
                    mb_idx * mini_batch_size : (mb_idx + 1) * mini_batch_size
                ]

                # Zero gradients
                for p in self.params:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                # Forward pass (with dropout for training)
                out = self._forward(xb, dropout=True)

                # Cost: negative log-likelihood from the last layer
                cost = self.layers[-1].cost_from_output(out, yb)

                # L2 regularization
                if lmbda != 0.0:
                    l2 = sum(
                        (layer.w**2).sum()
                        for layer in self.layers
                        if hasattr(layer, "w")
                    )
                    cost = cost + 0.5 * lmbda * l2 / num_training_batches

                # Backward pass and SGD update
                cost.backward()
                with torch.no_grad():
                    for p in self.params:
                        if p.grad is not None:
                            p.data -= eta * p.grad.data

                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [
                            self._accuracy(
                                validation_x[
                                    j * mini_batch_size : (j + 1) * mini_batch_size
                                ],
                                validation_y[
                                    j * mini_batch_size : (j + 1) * mini_batch_size
                                ],
                            )
                            for j in range(num_validation_batches)
                        ]
                    )
                    print(
                        "Epoch {0}: validation accuracy {1:.2%}".format(
                            epoch, validation_accuracy
                        )
                    )
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [
                                    self._accuracy(
                                        test_x[
                                            j * mini_batch_size : (j + 1)
                                            * mini_batch_size
                                        ],
                                        test_y[
                                            j * mini_batch_size : (j + 1)
                                            * mini_batch_size
                                        ],
                                    )
                                    for j in range(num_test_batches)
                                ]
                            )
                            print(
                                "The corresponding test accuracy is {0:.2%}".format(
                                    test_accuracy
                                )
                            )
        print("Finished training network.")
        print(
            "Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
                best_validation_accuracy, best_iteration
            )
        )
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))


#### Define layer types


class ConvPoolLayer:
    """A convolutional layer followed by 2×2 max pooling.

    `filter_shape` is (num_filters, num_input_maps, filter_h, filter_w).
    `image_shape`  is (mini_batch_size, num_input_maps, image_h, image_w).
    `poolsize`     is (pool_h, pool_w).
    """

    def __init__(
        self, filter_shape, image_shape, poolsize=(2, 2), activation_fn=sigmoid
    ):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn

        n_out = filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize)
        self.w = torch.nn.Parameter(
            torch.tensor(
                np.random.normal(0, np.sqrt(1.0 / n_out), size=filter_shape),
                dtype=torch.float32,
            )
        )
        self.b = torch.nn.Parameter(
            torch.tensor(
                np.random.normal(0, 1.0, size=(filter_shape[0],)),
                dtype=torch.float32,
            )
        )
        self.params = [self.w, self.b]

    def forward(self, x, dropout=False):  # conv layers never apply dropout
        x = x.reshape(-1, *self.image_shape)
        conv_out = F.conv2d(x, self.w)
        pooled = F.max_pool2d(conv_out, kernel_size=self.poolsize)
        return self.activation_fn(pooled + self.b.reshape(1, -1, 1, 1))


class FullyConnectedLayer:
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.w = torch.nn.Parameter(
            torch.tensor(
                np.random.normal(0.0, np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=torch.float32,
            )
        )
        self.b = torch.nn.Parameter(
            torch.tensor(
                np.random.normal(0.0, 1.0, size=(n_out,)),
                dtype=torch.float32,
            )
        )
        self.params = [self.w, self.b]

    def forward(self, x, dropout=False):
        x = x.reshape(-1, self.n_in)
        if dropout and self.p_dropout > 0:
            # Training: apply dropout to input, use full weights
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p_dropout))
            x = x * mask
        else:
            # Inference: scale weights by (1 - p_dropout) instead of dropping
            x = (1 - self.p_dropout) * x
        return self.activation_fn(x @ self.w + self.b)


class SoftmaxLayer:
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        self.w = torch.nn.Parameter(torch.zeros(n_in, n_out, dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.zeros(n_out, dtype=torch.float32))
        self.params = [self.w, self.b]

    def forward(self, x, dropout=False):
        x = x.reshape(-1, self.n_in)
        if dropout and self.p_dropout > 0:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p_dropout))
            x = x * mask
        else:
            x = (1 - self.p_dropout) * x
        return F.softmax(x @ self.w + self.b, dim=1)

    def cost_from_output(self, output, y):
        """Negative log-likelihood cost."""
        return F.nll_loss(torch.log(output + 1e-8), y)
