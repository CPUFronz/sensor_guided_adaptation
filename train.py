import math
import os
import time
from datetime import timedelta
from inspect import signature
from uuid import uuid4

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from finetune import FTModel
from sconv import SConvB
from utils import count_parameters
from utils import accuracy
from utils import DummyScheduler
from utils import fix_seeds
from utils import get_wandb_model
from constants import MODEL_DIR
from constants import WANDB_PROJECT


def train(dataloader, num_targets, model, device, mod_function, optim):
    train_loss = 0
    train_accuracy = 0

    model.train()
    has_aux = len(signature(model.forward).parameters) > 1

    for X, y in tqdm(dataloader, desc='Train', leave=False):
        X = X.to(device)
        XX, aux = mod_function(X)
        XX = XX.to(device)
        y = torch.nn.functional.one_hot(y, num_targets).float().to(device)

        if has_aux:
            aux = aux.to(device)
            pred = model(XX, aux)
        else:
            pred = model(XX)

        loss = optim['loss_fn'](pred, y)
        train_loss += loss.item()
        train_accuracy += accuracy(y, pred)

        optim['optimizer'].zero_grad()
        loss.backward()
        optim['optimizer'].step()
    optim['scheduler'].step()

    # average loss and accuracy
    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)
    return train_loss, train_accuracy


def test(dataloader, num_targets, model, device, mod_function, loss_fn):
    test_loss = 0
    test_accuracy = 0

    model.eval()
    has_aux = len(signature(model.forward).parameters) > 1

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Test', leave=False):
            X = X.to(device)
            XX, aux = mod_function(X)
            XX = XX.to(device)
            y = torch.nn.functional.one_hot(y, num_targets).float().to(device)

            if has_aux:
                aux = aux.to(device)
                pred = model(XX, aux)
            else:
                pred = model(XX)

            test_loss += loss_fn(pred, y).item()
            test_accuracy += accuracy(y, pred)

    # average loss and accuracy
    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)

    return test_loss, test_accuracy


def train_common(opt, model, mod_function, patience=100):
    train_loader = opt.train_loader
    test_loader = opt.test_loader
    targets = opt.n_targets
    loss_fn = opt.loss_function
    optimizer = opt.optimizer

    device = opt.device
    model = model.to(device)
    model.train()
    print(f'Start training of: {opt.model_name}')

    num_params = count_parameters(model)
    print('Trainable Parameters:', num_params)

    fname = f'models/{opt.model_name}_{str(uuid4().hex)[:8]}__last.pt'
    fname_best = fname.replace('__last', '__best')
    max_test_loss = np.Infinity
    best_epoch = 0

    model_optimizer = optimizer(model.parameters(), lr=opt.lr)

    optim = {
        'loss_fn': loss_fn,
        'optimizer': model_optimizer
    }
    optim['scheduler'] = opt.scheduler(optim['optimizer'], opt.epochs)

    print(f'Running on {device}')

    loop_start = time.time()
    for e in range(opt.epochs):
        train_loss, train_accuracy = train(train_loader, targets, model, device, mod_function, optim)
        test_loss, test_accuracy = test(test_loader, targets, model, device, mod_function, optim['loss_fn'])

        if mod_function.__name__ == 'mod_wrapper':
            # skip logging for fine-tuning
            pass
        else:
            log_dict = {
                'learning_rate': optim['scheduler'].get_last_lr()[0],
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy
            }
            wandb.log(log_dict)

        if test_loss < max_test_loss:
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model, fname_best)
            max_test_loss = test_loss
            best_epoch = e

        elapsed_time = time.time() - loop_start
        avg_iteration_time = elapsed_time / (e + 1)
        remaining_time = round(avg_iteration_time * (opt.epochs - e - 1))
        print(f'Epoch: {e:3d}   Train Loss: {train_loss:5.3f}   Test Loss: {test_loss:5.3f}   '
              f'Acc. Train: {train_accuracy:5.3f}   Acc. Test: {test_accuracy:5.3f}  '
              f'Best Epoch: {best_epoch:3d}  Remaining: {timedelta(seconds=remaining_time)}')

        if e - best_epoch > patience:
            print(f'Early stopping after {e} epochs')
            break

    torch.save(model, fname)
    last_model = wandb.Artifact(f'{opt.model_name}_last', type='model')
    last_model.add_file(fname)
    wandb.log_artifact(last_model)
    best_model = wandb.Artifact(f'{opt.model_name}_best', type='model')
    best_model.add_file(fname_best)
    wandb.log_artifact(best_model)

    return fname, fname_best


def train_ofa_common(opt, n_neurons, n_layers, loader, mod_fun, evaluations, alt_model=None):
    opt.model_name = opt.run_name.replace(':', '').replace(' ', '_').lower()

    opt.train_loader, opt.test_loader, opt.n_targets = loader(opt)
    n_channels = next(iter(opt.train_loader))[0].shape[1]

    if alt_model is None:
        model = SConvB(n_layers, n_neurons, n_channels, opt.n_targets, batchnorm=opt.batchnorm)
    else:
        model = alt_model
    n_params = count_parameters(model)

    add_hyper_params = {'n_params': n_params, 'n_layers': n_layers, 'n_neurons': n_neurons}
    config = {**vars(opt),  **add_hyper_params}

    with wandb.init(project=WANDB_PROJECT, config=config, name=opt.run_name, tags=opt.tags):
        wandb.run.log_code('.')

        _, model_path = train_common(opt, model, mod_fun)

        for evaluate in evaluations:
            model = torch.load(model_path, map_location=opt.device)
            evaluate['function'](opt, model)


def finetune_common(opt, fine_tune_values, loader, mod_fun, evaluations):
    base_model_fn, orig_opt = get_wandb_model(opt.base_run_id)

    for k, v in orig_opt.items():
        exec(f'opt.orig_{k} = "{v["value"]}"')
    opt.lr = float(opt.orig_lr)
    opt.seed = int(opt.orig_seed)
    opt.batch_size = int(opt.orig_batch_size)

    opt.scheduler = DummyScheduler
    with wandb.init(project=WANDB_PROJECT, config=opt, name=opt.run_name, tags=opt.tags):
        wandb.run.log_code('.')

        ft_models = {}
        for ftv in fine_tune_values:
            fix_seeds(opt.seed)

            opt.model_name = opt.run_name.replace(':', '').replace(' ', '_').lower() + f'_{ftv:.3f}'
            base_model = torch.load(base_model_fn, map_location=opt.device)
            opt.train_loader, opt.test_loader, opt.n_targets = loader(opt)

            def mod_wrapper(X):
                if mod_fun.__name__ == 'rotation':
                    return mod_fun(X, math.radians(ftv))
                else:
                    return mod_fun(X, ftv)

            opt.epochs = opt.epochs_fine_tune
            _, ft_mdl_fn = train_common(opt, base_model, mod_wrapper)
            ft_mdl = torch.load(ft_mdl_fn, map_location=opt.device)
            base_model = torch.load(base_model_fn, map_location=opt.device) # make sure that the original base model is used
            ft_models[f'{ftv:.3f}'] = ft_mdl

        if mod_fun.__name__ == 'rotation':
            ft_models['0.0'] = base_model
            ft_models['360.0'] = base_model
        elif mod_fun.__name__ == 'scale':
            ft_models['1.0'] = base_model

        for evaluate in evaluations:
            model = FTModel(ft_models, evaluate['transformation'])
            evaluate['function'](opt, model)
