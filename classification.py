import os
import argparse
import logging
import time
import numpy as np

import torch
import neptune
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.models import Model


run = neptune.init_run(
    project="10708/10708",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNWI1ZjhkZi1iN2I0LTQ0NDktOGEyNC1jMTY3Y2U3ZDRmMzEifQ==",
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--network", type=str, choices=["resnet", "odenet"], default="odenet"
)
parser.add_argument("--tol", type=float, default=1e-3)
parser.add_argument("--adjoint", type=eval, default=False, choices=[True, False])
parser.add_argument(
    "--downsampling-method", type=str, default="conv", choices=["conv", "res"]
)
parser.add_argument("--nepochs", type=int, default=160)
parser.add_argument("--data_aug", type=eval, default=True, choices=[True, False])
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=1000)

parser.add_argument("--save", type=str, default=".experiment1")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

run["parameters"] = args


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.ToTensor(),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_loader = DataLoader(
        datasets.MNIST(
            root=".data/mnist", train=True, download=True, transform=transform_train
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(
            root=".data/mnist", train=True, download=True, transform=transform_test
        ),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            root=".data/mnist", train=False, download=True, transform=transform_test
        ),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(
    batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates
):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == "__main__":
    makedirs(args.save)
    logger = get_logger(
        logpath=os.path.join(args.save, "logs"), filepath=os.path.abspath(__file__)
    )
    logger.info(args)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )

    is_odenet = args.network == "odenet"
    model = Model(is_odenet, args.downsampling_method).to(device)

    logger.info(model)
    logger.info("Number of parameters: {}".format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size,
        batch_denom=128,
        batches_per_epoch=batches_per_epoch,
        boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001],
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    print("Start training")
    for itr in tqdm(range(args.nepochs * batches_per_epoch)):
        print(1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = model.feature_layer.nfe
            model.feature_layer.nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = model.feature_layer.nfe
            model.feature_layer.nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0 or True:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save(
                        {"state_dict": model.state_dict(), "args": args},
                        os.path.join(args.save, "model.pth"),
                    )
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch,
                        batch_time_meter.val,
                        batch_time_meter.avg,
                        f_nfe_meter.avg,
                        b_nfe_meter.avg,
                        train_acc,
                        val_acc,
                    )
                )
