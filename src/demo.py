import argparse

import mnist_loader

LAYERS = [784, 30, 10]
EPOCHS = 30
MINI_BATCH_SIZE = 10
# default ETA = 0.5 for network2 module; for network module, use 3.0
ETA = 0.5
LMBDA = 5.0
MONITOR_EVALUATION_COST = False
MONITOR_EVALUATION_ACCURACY = True
MONITOR_TRAINING_COST = False
MONITOR_TRAINING_ACCURACY = False


def print_config():
    print(f"LAYERS                    = {LAYERS}")
    print(f"EPOCHS                    = {EPOCHS}")
    print(f"MINI_BATCH_SIZE           = {MINI_BATCH_SIZE}")
    print(f"ETA                       = {ETA}")
    print(f"LMBDA                     = {LMBDA}")
    print(f"MONITOR_EVALUATION_COST   = {MONITOR_EVALUATION_COST}")
    print(f"MONITOR_EVALUATION_ACCURACY = {MONITOR_EVALUATION_ACCURACY}")
    print(f"MONITOR_TRAINING_COST     = {MONITOR_TRAINING_COST}")
    print(f"MONITOR_TRAINING_ACCURACY = {MONITOR_TRAINING_ACCURACY}")


def run_network():
    import network

    training_data, _, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(LAYERS)
    net.SGD(training_data, EPOCHS, MINI_BATCH_SIZE, eta=ETA, test_data=test_data)


def run_network2():
    import network2

    training_data, validation_data, _ = mnist_loader.load_data_wrapper()
    net = network2.Network(LAYERS)
    net.SGD(
        training_data,
        EPOCHS,
        MINI_BATCH_SIZE,
        eta=ETA,
        lmbda=LMBDA,
        evaluation_data=validation_data,
        monitor_evaluation_cost=MONITOR_EVALUATION_COST,
        monitor_evaluation_accuracy=MONITOR_EVALUATION_ACCURACY,
        monitor_training_cost=MONITOR_TRAINING_COST,
        monitor_training_accuracy=MONITOR_TRAINING_ACCURACY,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "module",
        nargs="?",
        default="network2",
        choices=["network", "network2"],
    )
    args = parser.parse_args()
    print_config()
    if args.module == "network":
        run_network()
    else:
        run_network2()
