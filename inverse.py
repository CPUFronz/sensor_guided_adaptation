import time

import math
import json
import argparse
from glob import glob
from types import SimpleNamespace

import torch
import wandb

from torchvision.transforms import functional as TF

from sconv import SConvB
from utils import get_wandb_model
from utils import loaders_gtsrb
from utils import loaders_cifar
from evaluate import evaluate_scale
from evaluate import evaluate_rotation
from constants import WANDB_PROJECT


EVALS = [
    {'function': evaluate_scale, 'transformation': 'scale'},
    {'function': evaluate_rotation, 'transformation': 'rotation'},
]


class InverseWrapper(torch.nn.Module):
    def __init__(self, model, transformation):
        super().__init__()
        self.model = model
        self.transformation = transformation

    def forward(self, X, aux, measure_time=False):
        if measure_time:
            overhead_start = time.time()

        self.model.eval()
        if self.transformation == 'scale':
            X_proc = TF.affine(X, scale=1 / aux, angle=0, translate=[0, 0], shear=[0.0])
        elif self.transformation == 'rotation':
            X_proc = TF.rotate(X, -math.degrees(aux))
        else:
            raise ValueError(f'Unknown transformation {self.transformation}')

        # TODO: remove?
        if measure_time:
            overhead_time = time.time() - overhead_start
            inference_start = time.time()

        pred = self.model(X_proc)

        if measure_time:
            inference_time = time.time() - inference_start
            return pred, inference_time, overhead_time

        return self.model(X_proc)


def main(args):
    model_fn, orig_opt = get_wandb_model(args.id)

    opt = {k: v['value'] for k, v in orig_opt.items()}
    del opt['test_loader'], opt['train_loader'], opt['run_name']
    opt['base_run_name'] = orig_opt['run_name']['value']
    opt['base_run_id'] = args.id
    opt['method'] = args.method
    opt = SimpleNamespace(**opt)

    if 'GTSRB' in opt.base_run_name:
        _, opt.test_loader, _ = loaders_gtsrb(opt)
    elif 'CIFAR10' in opt.base_run_name:
        _, opt.test_loader, _ = loaders_cifar(opt)

    baseline_model = torch.load(model_fn, map_location=opt.device)

    if opt.method == 'rotation':
        eval = {'function': evaluate_rotation, 'transformation': 'rotation'}
    elif opt.method == 'scale':
        eval = {'function': evaluate_scale, 'transformation': 'scale'}

    opt.run_name = f'{opt.base_run_name.split(" " )[0]} {eval["transformation"].capitalize()} Inverse'
    with wandb.init(project=WANDB_PROJECT, config=opt, name=opt.run_name, tags=opt.tags):
        model = InverseWrapper(baseline_model, eval['transformation'])
        eval['function'](opt, model)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser('arguments for running inverse')
    parser.add_argument('--device', type=str, default=device, help='used device')
    parser.add_argument('id', type=str, help='wandb id to evaluate')
    parser.add_argument('method', type=str, help='which method to use for evaluation', choices=['rotation', 'scale'])
    args = parser.parse_args()

    main(args)
