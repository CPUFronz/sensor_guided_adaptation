from train import train_ofa_common
from transforms import rotation
from utils import fix_seeds
from utils import loaders_cifar
from utils import loaders_gtsrb
from utils import get_arguments

from constants import N_LAYERS


if __name__ == '__main__':
    datasets = ['cifar10', 'gtsrb']
    opt = get_arguments(datasets, use_different_n=True)
    opt.run_name = f'{opt.method.upper()} Rotation OFA_n-{opt.n}'
    opt.n_neurons = 2 ** opt.n
    opt.batchnorm = not opt.disable_batchnorm
    del opt.disable_batchnorm

    if opt.n == 6 or opt.n == 9:
        print('WARNING: n=6 and n=9 are done with OFA and OFA_large, retraining is not necessary')

    fix_seeds(opt.seed)
    loaders = loaders_cifar if opt.method == 'cifar10' else loaders_gtsrb
    train_ofa_common(opt, n_neurons=opt.n_neurons, n_layers=N_LAYERS, loader=loaders, mod_fun=rotation, evaluations=[])
