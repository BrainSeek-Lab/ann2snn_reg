import argparse
import os
import torch
import torch.nn as nn
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger
from Models.layer import IF

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j','--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b','--batch_size', default=300, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training.')
parser.add_argument('-suffix','--suffix', default='tiny_0.05', type=str, help='suffix')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
parser.add_argument('-data', '--dataset', default='tinyimagenet', type=str, help='dataset')
parser.add_argument('-arch','--model', default='vgg16', type=str, help='model')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-lr','--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-wd','--weight_decay', default=5e-4, type=float, help='weight_decay')
parser.add_argument('-dev','--device', default='0', type=str, help='device')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    seed_all(args.seed)
    train_loader, test_loader = datapool(args.dataset, args.batch_size)

    model = modelpool(args.model, args.dataset)
    model.set_L(args.L)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc = 0

    identifier = f"{args.model}_L[{args.L}]"
    if args.suffix:
        identifier += f"_{args.suffix}"

    log_dir = f"{args.dataset}-checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(os.path.join(log_dir, f"{identifier}.log"))
    logger.info("start training!")

    # QCFS regularizer weight (tune between 0.01 and 0.2)
    w_q = 0.05

    for epoch in range(args.epochs):
        if args.time > 0:
            for m in model.modules():
                if isinstance(m, IF):
                    m.reset_stats()
        task_loss, acc, qcfs_loss = train(
            model, device, train_loader, criterion,
            optimizer, args.time, w_q
        )
        logger.info(
            f"Epoch:[{epoch}/{args.epochs}]  "
            f"task_loss={task_loss:.5f}  "
            f"qcfs_loss={qcfs_loss:.6f}  "
            f"acc={acc:.3f}"
        )
        scheduler.step()

        tmp = val(model, test_loader, device, args.time)
        
    
        if args.time > 0:
            for m in model.modules():
                if isinstance(m, IF):
                    m.reset_stats()

        tmp = val(model, test_loader, device, args.time)
        logger.info(f"Epoch:[{epoch}/{args.epochs}]  Test acc={tmp:.3f}\n")
        if args.time > 0:
            for name, m in model.named_modules():
                if isinstance(m, IF):
                    rate = m.get_spike_rate()
                    logger.info(f"[{name}] spike_rate={rate:.4f}")
        if best_acc < tmp:
            best_acc = tmp
            torch.save(
                model.state_dict(),
                os.path.join(log_dir, f"{identifier}.pth")
            )

    logger.info(f"Best Test acc={best_acc:.3f}")

if __name__ == "__main__":
    main() 