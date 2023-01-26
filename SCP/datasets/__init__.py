from SCP.datasets.mnist import load_MNIST, load_MNIST_square, load_MNIST_C
from SCP.datasets.fashion_mnist import load_Fashion_MNIST
from SCP.datasets.cifar import load_CIFAR10, load_CIFAR10_BW, load_CIFAR100, load_CIFAR100_BW
from SCP.datasets.letters import load_MNIST_Letters
from SCP.datasets.kmnist import load_KMNIST
from SCP.datasets.not_mnist import load_notMNIST
from SCP.datasets.omniglot import load_omniglot
from SCP.datasets.flowers import load_flowers
from SCP.datasets.caltech import load_caltech101
from SCP.datasets.fgvc_aircraft import load_FGVCAircraft


in_distribution_datasets_loader = {
    'MNIST': load_MNIST,
    'Fashion_MNIST': load_Fashion_MNIST,
    'KMNIST': load_KMNIST,
    'Letters': load_MNIST_Letters,
    'MNIST_Square': load_MNIST_square,
    'CIFAR10': load_CIFAR10,
}

out_of_distribution_datasets_loader = {
    'MNIST': load_MNIST,
    'Fashion_MNIST': load_Fashion_MNIST,
    'KMNIST': load_KMNIST,
    'Letters': load_MNIST_Letters,
    'noMNIST': load_notMNIST,
    'omniglot': load_omniglot,
    'CIFAR10-BW': load_CIFAR10_BW,
    'MNIST-C': load_MNIST_C,
    'MNIST_Square': load_MNIST_square,
    'Flowers102': load_flowers,
    'FVGCAircraft': load_FGVCAircraft,
    'Caltech101': load_caltech101,
    'Food101': load_caltech101,
    'CIFAR100': load_CIFAR100,
}