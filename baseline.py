import torch

from train import train_ofa_common
from utils import main
from utils import fix_seeds
from utils import loaders_cifar
from utils import loaders_gtsrb
from evaluate import evaluate_scale
from evaluate import evaluate_rotation
from constants import N_NEURONS
from constants import N_LAYERS


EVAL_ROTATION = [{'function': evaluate_rotation, 'transformation': 'rotation'}]
EVAL_SCALE = [{'function': evaluate_scale, 'transformation': 'scale'}]


def dummy_mod(X):
    return X, torch.Tensor([0])


def train_cifar10_rotation(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'CIFAR10 Rotation Baseline'
    train_ofa_common(opt, n_neurons=N_NEURONS, n_layers=N_LAYERS, loader=loaders_cifar, mod_fun=dummy_mod, evaluations=EVAL_ROTATION)


def train_gtsrb_rotation(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'GTSRB Rotation Baseline'
    train_ofa_common(opt, n_neurons=N_NEURONS, n_layers=N_LAYERS, loader=loaders_gtsrb, mod_fun=dummy_mod, evaluations=EVAL_ROTATION)


def train_cifar10_scale(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'CIFAR10 Scale Baseline'
    train_ofa_common(opt, n_neurons=N_NEURONS, n_layers=N_LAYERS, loader=loaders_cifar, mod_fun=dummy_mod, evaluations=EVAL_SCALE)


def train_gtsrb_scale(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'GTSRB Scale Baseline'
    train_ofa_common(opt, n_neurons=N_NEURONS, n_layers=N_LAYERS, loader=loaders_gtsrb, mod_fun=dummy_mod, evaluations=EVAL_SCALE)


if __name__ == '__main__':
    methods = {
        'cifar10_rotation': train_cifar10_rotation,
        'gtsrb_rotation': train_gtsrb_rotation,
        'cifar10_scale': train_cifar10_scale,
        'gtsrb_scale': train_gtsrb_scale
    }
    main(methods)
