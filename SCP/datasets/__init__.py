from SCP.datasets.mnist import load_MNIST, load_MNIST_square, load_MNIST_C
from SCP.datasets.fashion_mnist import FashionMNIST
from SCP.datasets.cifar import CIFAR10, CIFAR10BW, CIFAR100, CIFAR100BW
from SCP.datasets.letters import Letters
from SCP.datasets.kmnist import KMNIST
from SCP.datasets.not_mnist import load_notMNIST
from SCP.datasets.omniglot import load_omniglot
from SCP.datasets.flowers import Flowers102
from SCP.datasets.caltech import Caltech101
from SCP.datasets.fgvc_aircraft import FGVCAircraft
from SCP.datasets.dtd import DTD
from SCP.datasets.genomics import load_oodgenomics
from SCP.datasets.celeb import CelebA
from SCP.datasets.fer2013 import FER2013
from SCP.datasets.gtsrb import GTSRB
from SCP.datasets.oxford_pets import load_oxford_pets
from SCP.datasets.eurosat import EuroSAT
from SCP.datasets.pcam import load_pcam

datasets_loader = {
    'MNIST': load_MNIST,
    'Fashion_MNIST': FashionMNIST,
    'KMNIST': KMNIST,
    'Letters': Letters,
    'noMNIST': load_notMNIST,
    'omniglot': load_omniglot,
    'CIFAR10-BW': CIFAR10BW,
    'MNIST-C': load_MNIST_C,
    'MNIST_Square': load_MNIST_square,
    'CIFAR10': CIFAR10,
    'Flowers102': Flowers102,
    'FGVCAircraft': FGVCAircraft,
    'Caltech101': Caltech101,
    'CIFAR100': CIFAR100,
    'DTD': DTD,
    'OODGenomics': load_oodgenomics,
    'CelebA': CelebA,
    'FER2013': FER2013,
    'GTSRB': GTSRB,
    'Oxford-pets': load_oxford_pets,
    'EuroSAT': EuroSAT,
    'PCAM': load_pcam,
}
