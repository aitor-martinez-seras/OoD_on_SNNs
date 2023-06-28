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
from SCP.datasets.gtsrb import GTSRB
from SCP.datasets.eurosat import EuroSAT
from SCP.datasets.tiny_imagenet import TinyImageNetLoader
from SCP.datasets.lsun import LSUN, LoaderLSUNResize, LoaderLSUNCrop, LoaderPatchesiSUN
from SCP.datasets.svhn import SVHN
from SCP.datasets.food import Food101
from SCP.datasets.neuromorphic_datasets import  NMNIST, NMNISTMissingEvents, CIFAR10DVS, NCALTECH101, DVSGesture, POKERDVS, DVSLip

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
    'Food101': Food101,
    'GTSRB': GTSRB,
    'EuroSAT': EuroSAT,
    'TinyImagenet': TinyImageNetLoader,
    'iSUN': LoaderPatchesiSUN,
    'LSUN_crop': LoaderLSUNCrop,
    'LSUN_resize': LoaderLSUNResize,
    'LSUN': LSUN,
    'SVHN': SVHN,
    'NMNIST': NMNIST,
    'CIFAR10DVS': CIFAR10DVS,
    'NMNIST_missing_events': NMNISTMissingEvents,
    'DVSGesture': DVSGesture,
    'POKERDVS': POKERDVS,
    'DVSLip': DVSLip,
    'NCALTECH101': NCALTECH101,
}
