from SCP.datasets.mnist import MNIST, MNIST_Square, MNIST_C_Loader
from SCP.datasets.fashion_mnist import FashionMNIST
from SCP.datasets.cifar import CIFAR10, CIFAR10BW, CIFAR100, CIFAR100BW
from SCP.datasets.letters import Letters
from SCP.datasets.kmnist import KMNIST
from SCP.datasets.not_mnist import notMNISTLoader
from SCP.datasets.omniglot import Omniglot
from SCP.datasets.flowers import Flowers102
from SCP.datasets.caltech import Caltech101
from SCP.datasets.fgvc_aircraft import FGVCAircraft
from SCP.datasets.dtd import DTD
from SCP.datasets.genomics import load_oodgenomics
from SCP.datasets.celeb import CelebA
from SCP.datasets.fer2013 import FER2013
from SCP.datasets.gtsrb import GTSRB
from SCP.datasets.oxford_pets import OxfordPets
from SCP.datasets.eurosat import EuroSAT
from SCP.datasets.pcam import PCAM
from SCP.datasets.tiny_imagenet import TinyImageNetLoader
from SCP.datasets.sun import SUN397

datasets_loader = {
    'MNIST': MNIST,
    'Fashion_MNIST': FashionMNIST,
    'KMNIST': KMNIST,
    'Letters': Letters,
    'noMNIST': notMNISTLoader,
    'omniglot': Omniglot,
    'CIFAR10-BW': CIFAR10BW,
    'MNIST-C': MNIST_C_Loader,
    'MNIST_Square': MNIST_Square,
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
    'Oxford-pets': OxfordPets,
    'EuroSAT': EuroSAT,
    'PCAM': PCAM,
    'TinyImagenet': TinyImageNetLoader,
    'SUN397': SUN397,
}
