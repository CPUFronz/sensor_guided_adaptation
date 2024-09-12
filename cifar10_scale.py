import numpy as np

from train import train_ofa_common
from train import finetune_common
from transforms import scale
from sconv import HHN_SConvB as SCN
from utils import main
from utils import fix_seeds
from utils import loaders_cifar
from evaluate import evaluate_scale
from constants import N_NEURONS
from constants import N_LAYERS
from constants import FINE_TUNE_VALUES_SCALE
from constants import SCALE_MIN
from constants import SCALE_MAX
from constants import D_SCALE


def train_ofa(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'CIFAR10 Scale OFA' if opt.nn_scale == 1 else f'CIFAR10 Scale OFA {opt.nn_scale}x'
    evals = [{'function': evaluate_scale, 'transformation': 'scale'}]
    train_ofa_common(opt, n_neurons=N_NEURONS * opt.nn_scale, n_layers=N_LAYERS, loader=loaders_cifar, mod_fun=scale, evaluations=evals)


def train_ft(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'CIFAR10 Scale Fine-Tune'
    evals = [{'function': evaluate_scale, 'transformation': 'scale'}]
    fine_tune_values = np.linspace(SCALE_MIN, SCALE_MAX, FINE_TUNE_VALUES_SCALE)
    finetune_common(opt, fine_tune_values, loaders_cifar, scale, evals)


def train_scn(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'CIFAR10 Scale SCN'
    evals = [{'function': evaluate_scale, 'transformation': 'scale'}]
    # not the most elegant way, but let's just reuse the function for OFA for the SCN model
    opt.D = D_SCALE if opt.D is None else opt.D
    model = SCN(hin=1, dimensions=opt.D, n_layers=N_LAYERS, n_units=opt.inference_neurons, n_channels=3, n_classes=10, use_batchnorm=opt.batchnorm, device=opt.device)
    train_ofa_common(opt, opt.inference_neurons, N_LAYERS, loader=loaders_cifar, mod_fun=scale, evaluations=evals, alt_model=model)


if __name__ == '__main__':
    methods = {
        'ofa': train_ofa,
        'ft': train_ft,
        'scn': train_scn
    }
    main(methods)
