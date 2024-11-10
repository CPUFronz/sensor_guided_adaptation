import os
import re
import math
import json
import argparse
from inspect import signature
from types import SimpleNamespace

import numpy as np
import torch

from utils import accuracy
from utils import get_wandb_model
from utils import loaders_gtsrb
from utils import loaders_cifar
from utils import fix_seeds
from transforms import scale, rotation
from inverse import InverseWrapper
from constants import SCALE_MIN
from constants import SCALE_MAX
from constants import NUM_WORKERS
from constants import BATCH_SIZE

DEFAULT_NOISE_PERCENTAGES = [0.5, 1, 2, 5, 10]
PATH = 'runs/{}'


def process_batch(X, y, model, has_aux, mod_fn, factor, noise_percentage, device, n_classes):
    y = torch.nn.functional.one_hot(y, n_classes).float().to(device)
    XX, aux = mod_fn(X, factor, sensor_noise_percentage=noise_percentage)
    XX = XX.to(device)

    if has_aux:
        aux = aux.to(device)
        y_hat = model(XX, aux)
    else:
        y_hat = model(XX)

    return accuracy(y, y_hat)


def main(args):
    model_fn, opt = get_wandb_model(args.id)
    if 'Inverse' in opt['run_name']['value']:
        model_fn, _ = get_wandb_model(opt['base_run_id']['value'])
        model = torch.load(model_fn, map_location=args.device)
        transformation = opt['run_name']['value'].split(' ')[1].lower()
        model = InverseWrapper(model, transformation)
    else:
        model = torch.load(model_fn, map_location=args.device)
        model = model.to(args.device)
        model.device = args.device

    opt = {k: v['value'] for k, v in opt.items()}
    opt = SimpleNamespace(**opt)
    # overwrite the following values
    opt.batch_size = args.batch_size
    opt.num_workers = args.num_workers

    run_name = opt.run_name

    match = re.match(r"(\w+)\s+(\w+)\s+(\w+.*)", run_name)
    if match:
        dataset, transformation, method = match.groups()

    print(f'Running evaluation for {run_name}')

    # use quasi-fixed factors
    fix_seeds(opt.seed)
    if 'GTSRB' in run_name:
        _, test_loader, n_classes  = loaders_gtsrb(opt)
    elif 'CIFAR10' in run_name:
        _, test_loader, n_classes = loaders_cifar(opt)

    if 'Scale' in run_name:
        mod_fn = scale
        factors = np.random.uniform(SCALE_MIN, SCALE_MAX, len(test_loader))
    elif 'Rotation' in run_name:
        mod_fn = rotation
        factors = np.random.uniform(0, 2 * math.pi, len(test_loader))

    has_aux = len(signature(model.forward).parameters) > 1

    results = {}
    noise_levels = [0] + args.noise
    for p in noise_levels:
        print(f'Noise percentage: {p}')

        acc = 0
        for idx, (X, y) in enumerate(test_loader):
            batch_accuracy = process_batch(X, y, model, has_aux, mod_fn, factors[idx], p, args.device, n_classes)
            acc += batch_accuracy

        acc /= len(test_loader)
        results[p] = acc

    fn = os.path.join(PATH, dataset + '__' + transformation + '__' + method + f'__{opt.seed}.json')
    with open(fn, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser('arguments for evaluating on noisy data')
    parser.add_argument('--device', type=str, default=device, help='used device')
    parser.add_argument('--noise', nargs='+', type=float, help='noise percentages', default=DEFAULT_NOISE_PERCENTAGES)
    parser.add_argument('--num_workers', type=int, help='number of workers for data loader', default=NUM_WORKERS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size for evaluating')
    parser.add_argument('id', type=str, help='wandb id to evaluate')
    args = parser.parse_args()

    os.makedirs(PATH, exist_ok=True)

    main(args)
