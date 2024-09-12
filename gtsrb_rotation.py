from train import train_ofa_common
from train import finetune_common
from utils import main
from utils import fix_seeds
from utils import loaders_gtsrb
from transforms import rotation
from evaluate import evaluate_rotation
from sconv import HHN_SConvB as SCN
from constants import N_NEURONS
from constants import N_LAYERS
from constants import FINE_TUNE_INTERVAL_ROTATION
from constants import D_ROTATION


def train_ofa(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'GTSRB Rotation OFA' if opt.nn_scale == 1 else f'GTSRB Rotation OFA {opt.nn_scale}x'
    evals = [{'function': evaluate_rotation, 'transformation': 'rotate'}]
    train_ofa_common(opt, n_neurons=N_NEURONS * opt.nn_scale, n_layers=N_LAYERS, loader=loaders_gtsrb, mod_fun=rotation, evaluations=evals)


def train_ft(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'GTSRB Rotation Fine-Tune'
    evals = [{'function': evaluate_rotation, 'transformation': 'rotate'}]
    fine_tune_values = list(range(FINE_TUNE_INTERVAL_ROTATION, 360, FINE_TUNE_INTERVAL_ROTATION))
    finetune_common(opt, fine_tune_values=fine_tune_values, loader=loaders_gtsrb, mod_fun=rotation, evaluations=evals)


def train_scn(opt):
    fix_seeds(opt.seed)
    opt.run_name = 'GTSRB Rotation SCN'
    evals = [{'function': evaluate_rotation, 'transformation': 'rotate'}]
    # not the most elegant way, but let's just reuse the function for OFA for the SCN model
    opt.D = D_ROTATION if opt.D is None else opt.D
    model = SCN(hin=1, dimensions=opt.D, n_layers=N_LAYERS, n_units=opt.inference_neurons, n_channels=3, n_classes=43, device=opt.device)
    train_ofa_common(opt, opt.inference_neurons, N_LAYERS, loader=loaders_gtsrb, mod_fun=rotation, evaluations=evals, alt_model=model)


if __name__ == '__main__':
    methods = {
        'ofa': train_ofa,
        'ft': train_ft,
        'scn': train_scn
    }
    main(methods)
