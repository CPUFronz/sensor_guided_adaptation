from inspect import signature

import numpy as np
import torch
import wandb
from tqdm import tqdm

from utils import accuracy
from utils import loaders_cifar, loaders_gtsrb
from transforms import scale, rotation

from constants import SCALE_MIN, SCALE_MAX, NUM_EVAL_SCALE_FACTORS


def process_batch(X, y, model, has_aux, mod_fn, mod_param, device, n_classes):
    y = torch.nn.functional.one_hot(y, n_classes).float().to(device)
    XX, aux = mod_fn(X, mod_param)
    XX = XX.to(device)

    if has_aux:
        aux = aux.to(device)
        y_hat = model(XX, aux)
    else:
        y_hat = model(XX)

    return accuracy(y, y_hat)


def evaluate_scale(opt, model, num_scale_factors=NUM_EVAL_SCALE_FACTORS):
    scale_factors = np.linspace(SCALE_MIN, SCALE_MAX, num_scale_factors)
    # add 1. to scale factors, to show how well the method works for the original image
    if 1. not in scale_factors:
        scale_factors = np.append(1., scale_factors)
        scale_factors.sort()

    test_loader = opt.test_loader
    n_classes = opt.n_targets

    model.eval()
    with torch.no_grad():
        has_aux = len(signature(model.forward).parameters) > 1
        for factor in tqdm(scale_factors):
            acc = 0
            for X, y in test_loader:
                acc += process_batch(X, y, model, has_aux, scale, factor, opt.device, n_classes)
            acc /= len(test_loader)
            wandb.log({'Accuracy per Scale Factor': acc, 'Scale Factor': factor})


def evaluate_rotation(opt, model):
    test_loader = opt.test_loader
    n_classes = opt.n_targets

    model.eval()
    with torch.no_grad():
        has_aux = len(signature(model.forward).parameters) > 1
        for angle in tqdm(range(360)):
            acc = 0
            angle_r = np.radians(angle)
            for X, y in test_loader:
                acc += process_batch(X, y, model, has_aux, rotation, angle_r, opt.device, n_classes)

            acc /= len(test_loader)
            wandb.log({'Accuracy per Angle': acc, 'Angle': angle})
