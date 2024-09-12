import json
import random
import argparse
from glob import glob

import wandb
import torch
import numpy as np

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine

from constants import SEED
from constants import BATCH_SIZE
from constants import LR
from constants import EPOCHS
from constants import IMG_SIZE
from constants import EPOCHS_FINE_TUNE
from constants import WANDB_PROJECT
from constants import NUM_WORKERS
from constants import N_NEURONS


def fix_seeds(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(target, predicted):
    y_true = target.argmax(axis=1).detach().cpu().numpy()
    y_pred = predicted.argmax(axis=1).detach().cpu().numpy()
    return np.sum(np.equal(y_true, y_pred)) / len(y_true)


def get_arguments(train_methods, use_different_n=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser('arguments for training')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size for training and testing')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of training epochs')
    parser.add_argument('--device', type=str, default=device, help='used device')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE, help='size of image NxN')
    parser.add_argument('--tags', nargs='+', default=[], help='list of tags for the run')
    parser.add_argument('--seed', type=int, default=SEED, help='seed for the run')
    parser.add_argument('--disable_batchnorm', action='store_true', help='disable batchnorm for model')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use for training')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR', help='learning rate scheduler to use for training')
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss', help='loss function to use for training')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='number of workers for data loader')
    parser.add_argument('--nn_scale', type=int, default=1, help='scale factor for the width of the neural network (only for OFA)')
    parser.add_argument('--base_run_id', type=str, help='wandb id of the baseline run to use for fine-tuning (only for fine-tuning)')
    parser.add_argument('--epochs_fine_tune', type=int, default=EPOCHS_FINE_TUNE, help='number of epochs to fine tune (only for fine-tuning)')
    parser.add_argument('--D', type=int, default=None, help='dimensions of SCN configuration subspace (only for SCN)')
    parser.add_argument('--inference_neurons', type=int, default=N_NEURONS, help='number of neurons for a inference network (only for SCN)')
    parser.add_argument('method', type=str, help='methods to use for training', choices=list(train_methods))

    if use_different_n:
        parser.add_argument('n', type=int, help='exponent for neurons in layer (2^n)')

    args = parser.parse_args()
    args.optimizer = eval(f'torch.optim.{args.optimizer}')
    args.scheduler = eval(f'torch.optim.lr_scheduler.{args.scheduler}')
    args.loss_function = eval(f'torch.nn.{args.loss_function}()')

    return args


def loaders(opt, train_data, test_data, shuffle):
    fix_seeds(opt.seed)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=shuffle, drop_last=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, drop_last=True, num_workers=opt.num_workers)

    # get number of targets from data loader
    tmp_train_loader = DataLoader(test_data, batch_size=len(train_data))
    for _, y_train in tmp_train_loader: break
    num_targets = len(set([y.item() for y in y_train]))

    return train_loader, test_loader, num_targets


def loaders_cifar(opt, shuffle=True):
    transform = Compose([
        Resize((opt.img_size, opt.img_size)),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )
    return loaders(opt, train_data, test_data, shuffle)


def loaders_gtsrb(opt, shuffle=True):
    # scale factor to normalize images, 1.5 showed to have the best results
    scale_factor = 1.5

    transform = Compose([
        Resize((opt.img_size, opt.img_size)),
        ToTensor(),
        Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
        RandomAffine(degrees=(0, 0), scale=(scale_factor, scale_factor))
    ])

    train_data = datasets.GTSRB(
        root="data",
        split='train',
        download=True,
        transform=transform,
    )
    test_data = datasets.GTSRB(
        root="data",
        split='test',
        download=True,
        transform=transform
    )
    return loaders(opt, train_data, test_data, shuffle)


def get_wandb_model(wandb_id):
    run = wandb.Api().run(f'{WANDB_PROJECT}/{wandb_id}')

    artifacts = run.logged_artifacts()
    model_fn = ''
    for a in artifacts:
        if a.type == 'model' and 'best' in a.name:
            model_fn = glob(a.download() + '/*.pt')[0]
            break

    opt = json.loads(run.json_config)
    opt['run_name']['value'] = run.name # use current run name instead of the (maybe old) one from the config

    return model_fn, opt


def main(methods):
    opt = get_arguments(methods.keys())
    opt.batchnorm = not opt.disable_batchnorm
    del opt.disable_batchnorm

    # TODO: remove?
    if opt.method == 'ft':
        assert opt.base_run_id is not None, 'need a base_run_id from a baseline run for fine-tuning'

    print(f'Starting training of {opt.method}')
    methods[opt.method](opt)


class DummyScheduler():
    def __init__(self, optimizer, epochs):
        self.is_dummy = True

    def step(self):
        return


if __name__ == '__main__':
    pass