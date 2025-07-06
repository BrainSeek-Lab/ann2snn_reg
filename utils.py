import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import random
import os
import logging
from Models.layer import IF

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
#QCFS regularizer:
def qcfs_quantization_loss(z, lambda_param, L, phi=0.5):
  
    c = (z * L / lambda_param) + phi
    j = torch.round(c)
    q = (j - phi) / L * lambda_param
    return ((z - q) ** 2).mean()

def train(model, device, train_loader, criterion, optimizer, T, w_q=0.0):
    running_loss = 0.0
    total = 0
    correct = 0
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).mean(0) if T > 0 else model(images)
        loss_task = criterion(outputs, labels)

        # QCFS regularizer across all IF layers
        zs = []
        for m in model.modules():
            if isinstance(m, IF) and hasattr(m, 'last_mem'):
                zs.append(m.last_mem.view(m.last_mem.size(0), -1))
        if zs:
            z_all = torch.cat(zs, dim=1)  # [batch, total_neurons]
            first_if = next(m for m in model.modules() if isinstance(m, IF))
            lambda_param = first_if.thresh
            L = first_if.L
            qcfs_reg = qcfs_quantization_loss(z_all, lambda_param, L, phi=0.5)
        else:
            qcfs_reg = torch.tensor(0.0, device=device)

        loss = loss_task + w_q * qcfs_reg
        loss.backward()
        optimizer.step()

        running_loss += loss_task.item()
        total += labels.size(0)
        _, preds = outputs.cpu().max(1)
        correct += preds.eq(labels.cpu()).sum().item()

    avg_loss = running_loss / len(train_loader)
    acc = 100.0 * correct / total
    return avg_loss, acc,qcfs_reg.item()

def val(model, test_loader, device, T):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).mean(0) if T > 0 else model(inputs)
            _, preds = outputs.cpu().max(1)
            total += targets.size(0)
            correct += preds.eq(targets.cpu()).sum().item()
    return 100.0 * correct / total
