from .getdataloader import *
from torch.utils.data import DataLoader

def datapool(DATANAME, batchsize, num_workers=4, pin_memory=True):
    if DATANAME.lower() == 'cifar10':
        train_loader, test_loader = GetCifar10(batchsize)
    elif DATANAME.lower() == 'cifar100':
        train_loader, test_loader = GetCifar100(batchsize)
    elif DATANAME.lower() == 'imagenet':
        train_loader, test_loader = GetImageNet(batchsize)
    elif DATANAME.lower() == 'tinyimagenet':
        train_loader, test_loader = GetTinyImageNet(batchsize)
    else:
        print("Still not support this model")
        exit(0)
    return train_loader, test_loader