from .market import Market
from .import utils
from .base import BaseDataset


_type = {
    'market': Market
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
