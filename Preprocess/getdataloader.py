from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch, os
from Preprocess.augment import Cutout, CIFAR10Policy

import urllib.request
import zipfile
import shutil

# _file_ is Preprocess/getdataloader.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DIR = {
    'CIFAR10': os.path.join(PROJECT_ROOT, '..', 'data', 'CIFAR10'),
    'CIFAR100': os.path.join(PROJECT_ROOT, '..', 'data', 'CIFAR100'),
    'ImageNet': os.path.join(PROJECT_ROOT, '..', 'data', 'ImageNet'),
    'TINY_IMAGENET_ROOT': os.path.join(PROJECT_ROOT, '..', 'data', 'TinyImageNet')
}

def download_and_prepare_tiny_imagenet(root):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    extract_path = os.path.join(root, "tiny-imagenet-200")
    final_path = os.path.join(root, "TinyImageNet")

    if not os.path.exists(final_path):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)
        os.rename(extract_path, final_path)
        os.remove(zip_path)
        print("Tiny ImageNet downloaded and extracted.")

        # Reorganize val images into subfolders
        val_dir = os.path.join(final_path, "val")
        images_dir = os.path.join(val_dir, "images")
        anno_file = os.path.join(val_dir, "val_annotations.txt")
        with open(anno_file, "r") as f:
            for line in f:
                img, cls = line.split()[:2]
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(images_dir, img)
                dst = os.path.join(cls_dir, img)
                if os.path.exists(src):
                    shutil.move(src, dst)
        shutil.rmtree(images_dir)
        print("Validation images reorganized for ImageFolder.")
    else:
        print("Tiny ImageNet already exists.")

def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=1, length=16)
    ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    root = DIR['CIFAR10']
    train_data = datasets.CIFAR10(root, train=True,
                                  transform=trans_t,
                                  download=True)
    test_data  = datasets.CIFAR10(root, train=False,
                                  transform=trans,
                                  download=True)

    train_loader = DataLoader(train_data, batch_size=batchsize,
                              shuffle=True, num_workers=8)
    test_loader  = DataLoader(test_data,  batch_size=batchsize,
                              shuffle=False, num_workers=8)
    return train_loader, test_loader

def GetCifar100(batchsize):
    trans_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[n/255. for n in [129.3, 124.1, 112.4]],
            std =[n/255. for n in [68.2,  65.4,  70.4]]
        ),
        Cutout(n_holes=1, length=16)
    ])
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[n/255. for n in [129.3, 124.1, 112.4]],
            std =[n/255. for n in [68.2,  65.4,  70.4]]
        )
    ])

    root = DIR['CIFAR100']
    train_data = datasets.CIFAR100(root, train=True,
                                   transform=trans_t,
                                   download=True)
    test_data  = datasets.CIFAR100(root, train=False,
                                   transform=trans,
                                   download=True)

    train_loader = DataLoader(train_data, batch_size=batchsize,
                              shuffle=True, num_workers=8, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=batchsize,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def GetTinyImageNet(batchsize, num_workers=8, pin_memory=True, attack=False):
    from .augment import Cutout, ImageNetPolicy

    root = DIR['TINY_IMAGENET_ROOT']
    download_and_prepare_tiny_imagenet(os.path.dirname(root))

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.480,0.448,0.398], std=[0.277,0.269,0.282]),
        Cutout(n_holes=1, length=16)
    ])
    if attack:
        test_tf = transforms.Compose([transforms.ToTensor()])
    else:
        test_tf = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.480,0.448,0.398], std=[0.277,0.269,0.282]),
        ])

    train_ds = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(root, 'val'),   transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batchsize, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader