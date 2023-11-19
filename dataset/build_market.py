from .utils import build_transforms, train_collate_fn, val_collate_fn
from .market1501 import Market1501
from .bases import ImageDataset
from torch.utils.data import DataLoader

def build_market_dataloader():
    train_transforms = build_transforms(is_train=True)
    val_transforms = build_transforms(is_train=False)
    num_workers = 8
    dataset = Market1501('../ITCS-5145-CV/learning/')
    
    train = dataset.train
    num_train_pids = dataset.num_train_pids
    num_train_cams = dataset.num_train_cams
    
    num_classes = dataset.num_train_pids
    
    train_set = ImageDataset(dataset.train, train_transforms)
    
    train_loader = DataLoader(
                    train_set, batch_size=50, shuffle=True, num_workers=num_workers,
                    collate_fn=train_collate_fn
    )
    
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    
    val_loader = DataLoader(
                val_set, batch_size=50, shuffle=False, num_workers=num_workers,
                collate_fn=val_collate_fn
            )
    
    num_query = len(dataset.query)
    val_loader_market = {
            'val_loader': val_loader,
            'num_query': num_query
            }

    return train_loader, val_loader_market, num_classes 