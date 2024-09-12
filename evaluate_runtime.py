# this script is used to evaluate the runtime of the model on the edge device
# it requires a different environment because of ai_edge_torch:
#       pip install -r https://github.com/google-ai-edge/ai-edge-torch/releases/download/v0.1.1/requirements.txt
#       pip install ai-edge-torch==0.1.1
# also adb is required to be installed

import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ai_edge_torch

from utils import get_wandb_model

PUSH_CMD = 'adb push tflite/{} /data/local/tmp/'
BENCHMARK_CMD = 'adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity --es args \'"--graph=/data/local/tmp/{} --num_threads=1"\''

def convert(mdl, sample_inputs, fn_out, check_model=True):
    torch_output = mdl(*sample_inputs)
    edge_model = ai_edge_torch.convert(mdl.eval(), sample_inputs)

    edge_output = edge_model(*sample_inputs)

    if check_model:
        if (np.allclose(
                torch_output.detach().numpy(),
                edge_output,
                atol=1e-5,
                rtol=1e-5,
        )):
            print("Inference result with Pytorch and TfLite was within tolerance")
        else:
            print("Something wrong with Pytorch --> TfLite")

    edge_model.export('tflite/' + fn_out)

class HyperModel(torch.nn.Module):
    def __init__(self, hyper_stack):
        super(HyperModel, self).__init__()
        self.hyper_stack = hyper_stack

    def forward(self, hyper_x):
        return self.hyper_stack(hyper_x)

class Configuration(torch.nn.Module):
    def __init__(self, n_channels, n_units, dimensions, n_classes=10):
        super(Configuration, self).__init__()

        self.n_channels = n_channels
        self.n_units = n_units
        self.dimensions = dimensions
        self.n_classes = n_classes

        # conv0 = inital convolution
        # conv1 = next convolution, number of convolutional layers is supposed to by dynamic, but for this it is fixed, to keep things easier for TensorFlow Lite conversation
        # convF = final convolution
        # fc    = fully connected final layer
        self.weight_list_conv0, self.bias_list_conv0 = self.create_param_combination_conv(in_channels=n_channels, out_channels=n_units, kernel=9)
        self.weight_list_conv1, self.bias_list_conv1 = self.create_param_combination_conv(in_channels=n_units, out_channels=n_units, kernel=3)
        self.weight_list_convF, self.bias_list_convF = self.create_param_combination_conv(in_channels=n_units, out_channels=n_units, kernel=13)
        self.weight_list_fc, self.bias_list_fc = self.create_param_combination_linear(in_features=n_units, out_features=n_classes)

    def calculate_weighted_sum(self, param_list, factors):
        # weighted_list = [a * b for a, b in zip(param_list, factors)]
        # return torch.sum(torch.stack(weighted_list), dim=0)
        # optimized version of the original code above:
        param_tuple = tuple(param_list)
        stacked_params = torch.stack(param_tuple)
        factors_tensor = torch.tensor(factors)
        while len(factors_tensor.shape) < len(stacked_params.shape):
            factors_tensor = factors_tensor.unsqueeze(-1)
        weighted_sum = torch.sum(stacked_params * factors_tensor, dim=0)
        return weighted_sum

    def create_param_combination_conv(self, in_channels, out_channels, kernel):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel, kernel)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = nn.Parameter(torch.empty(out_channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return weight_list, bias_list

    def create_param_combination_linear(self, in_features, out_features):
        weight_list = nn.ParameterList()
        bias_list = nn.ParameterList()
        for _ in range(self.dimensions):
            weight = nn.Parameter(torch.empty((out_features, in_features)))
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight_list.append(weight)

            bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            bias_list.append(bias)
        return weight_list, bias_list

    def forward(self, hyper_output):
        weights = {}
        biases = {}

        weights['w_conv0'] = self.calculate_weighted_sum(self.weight_list_conv0, hyper_output)
        biases['b_conv0'] = self.calculate_weighted_sum(self.bias_list_conv0, hyper_output)

        weights['w_conv1'] = self.calculate_weighted_sum(self.weight_list_conv1, hyper_output)
        biases['b_conv1'] = self.calculate_weighted_sum(self.bias_list_conv1, hyper_output)

        weights['w_convF'] = self.calculate_weighted_sum(self.weight_list_convF, hyper_output)
        biases['b_convF'] = self.calculate_weighted_sum(self.bias_list_convF, hyper_output)

        weights['w_fc'] = self.calculate_weighted_sum(self.weight_list_fc, hyper_output)
        biases['b_fc'] = self.calculate_weighted_sum(self.bias_list_fc, hyper_output)

        return weights, biases

class InferenceModel(torch.nn.Module):
    def __init__(self, n_units, weigths, biases):
        super(InferenceModel, self).__init__()

        self.n_units = n_units

        self.weights = weigths
        self.biases = biases

        self.batch_norm1 = nn.BatchNorm2d(weigths['w_conv1'].shape[0])

    def forward(self, x):
        # first convolutional layer
        logits = F.conv2d(x, weight=self.weights['w_conv0'], bias=self.biases['b_conv0'], stride=2, padding=1)
        logits = torch.relu(logits)

        # second convolutional layer
        logits = F.conv2d(logits, weight=self.weights['w_conv1'], bias=self.biases['b_conv1'], stride=1, padding=1)
        logits = self.batch_norm1(logits)
        logits = torch.relu(logits)

        # final convolutional layer
        logits = F.conv2d(logits, weight=self.weights['w_convF'], bias=self.biases['b_convF'], stride=1, padding=0)
        logits = torch.relu(logits)
        logits = torch.flatten(logits, start_dim=1)
        logits = F.linear(logits, weight=self.weights['w_fc'], bias=self.biases['b_fc'])

        return logits


class InverseModel(torch.nn.Module):
    def __init__(self):
        super(InverseModel, self).__init__()
        self.X = torch.randn(1, 3, 32, 32)

    def forward(self, angle):
        N, C, H, W = self.X.shape

        angle = torch.tensor(angle)
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        # R = torch.Tensor([[cos_theta, -sin_theta, 0],
        #                  [sin_theta, cos_theta, 0]])
        # proper rotation is not able to be converted to TensorFlow Lite, just use ones instead
        R = torch.ones(2, 3)

        theta = R.unsqueeze(0).repeat(N, 1, 1)
        grid = F.affine_grid(theta, self.X.size())
        rotated_tensor = F.grid_sample(self.X, grid, align_corners=True)

        return rotated_tensor


if __name__ == '__main__':
    os.makedirs('tflite', exist_ok=True)
    os.system('adb logcat -c')
    os.system('adb shell am force-stop org.tensorflow.lite.benchmark')
    os.system('adb shell rm /data/local/tmp/*.tflite')

    parser = argparse.ArgumentParser('arguments for evaluating runtime')
    parser.add_argument('mode', type=str, help='type of model to evaluate', choices=['base', 'inverse', 'scn'])
    parser.add_argument('ids', type=str, help='wandb ids to evaluate', nargs='+')
    args = parser.parse_args()

    for i in args.ids:
        model_fn, opt = get_wandb_model(i)
        model_fn_base = os.path.basename(model_fn)
        seed = opt['seed']['value']

        if args.mode == 'base':
            model = torch.load(model_fn, map_location='cpu')
            sample_inputs = (torch.randn(1, 3, 32, 32),)
            fn_out = f'{model_fn_base}__{seed}.tflite'
            convert(model, sample_inputs, fn_out)
            print(BENCHMARK_CMD.format(fn_out))
            os.system(PUSH_CMD.format(fn_out))
            os.system(BENCHMARK_CMD.format(fn_out))

        elif args.mode == 'inverse':
            os.system(f'adb shell log -t tflite  "Inference timings Inverse:"')

            inverse_model = InverseModel()
            sample_inputs_inv = (torch.randn(1),)
            fn_out_inv = f'inverse_preprocess__{seed}.tflite'
            convert(inverse_model, sample_inputs_inv, fn_out_inv)
            os.system(PUSH_CMD.format(fn_out_inv))
            os.system(BENCHMARK_CMD.format(fn_out_inv))

            orig_model_fn, _ = get_wandb_model(opt['base_run_id']['value'])
            base_model = torch.load(orig_model_fn, map_location='cpu')
            sample_inputs_base = (torch.randn(1, 3, 32, 32),)
            fn_out_base = f'inverse_inference__{seed}.tflite'
            convert(base_model, sample_inputs_base, fn_out_base)
            os.system(PUSH_CMD.format(fn_out_base))
            os.system(BENCHMARK_CMD.format(fn_out_base))

        elif args.mode == 'scn':
            os.system(f'adb shell log -t tflite  "Inference timings SCN:"')
            model = torch.load(model_fn, map_location='cpu')

            # hyper model
            hyper_model = HyperModel(model.hyper_stack)
            sample_inputs_hyper = (torch.randn(1),)
            fn_out_hyper = f'scn_hyper__{seed}.tflite'
            convert(hyper_model, sample_inputs_hyper, fn_out_hyper)
            os.system(PUSH_CMD.format(fn_out_hyper))
            os.system(BENCHMARK_CMD.format(fn_out_hyper))

            # configuration model
            configuration = Configuration(model.n_channels, model.n_units, model.dimensions, model.n_classes)
            sample_inputs_configuration = (torch.randn(model.dimensions),)
            fn_out_configuration = f'scn_configuration__{seed}.tflite'
            convert(configuration, sample_inputs_configuration, fn_out_configuration, check_model=False)
            os.system(PUSH_CMD.format(fn_out_configuration))
            os.system(BENCHMARK_CMD.format(fn_out_configuration))

            # inference model
            weights, biases = configuration(*sample_inputs_configuration)
            inference_model = InferenceModel(model.n_units, weights, biases)
            sample_inputs_inference = (torch.randn(1, 3, 32, 32),)
            fn_out_inference = f'scn_inference__{seed}.tflite'
            convert(inference_model, sample_inputs_inference, fn_out_inference)
            os.system(PUSH_CMD.format(fn_out_inference))
            os.system(BENCHMARK_CMD.format(fn_out_inference))

        time.sleep(1)
