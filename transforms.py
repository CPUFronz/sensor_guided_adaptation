import math

import numpy as np
import torch
from torchvision.transforms import functional as TF
from constants import SCALE_MIN
from constants import SCALE_MAX


def scale(X, factor=None, sensor_noise_percentage=None):
    if factor is None:
        factor = np.random.uniform(SCALE_MIN, SCALE_MAX)

    noise = 0
    if sensor_noise_percentage is not None:
        noise_amplitude = sensor_noise_percentage / 100.0 * (SCALE_MAX - SCALE_MIN)
        noise = np.random.normal(0, noise_amplitude)
    noisy_sensor = max(SCALE_MIN, min(factor + noise, SCALE_MAX))

    XX = TF.affine(X, scale=factor, angle=0, translate=[0, 0], shear=[0.0])

    return XX, torch.Tensor([noisy_sensor])


def rotation(X, angle=None, sensor_noise_percentage=None):
    if angle is None:
        angle = np.random.uniform(0, 2 * math.pi)

    noise = 0
    if sensor_noise_percentage is not None:
        noise_amplitude = sensor_noise_percentage / 100.0 * (2 * math.pi - 0)
        noise = np.random.normal(0, noise_amplitude)
    noisy_sensor = max(0, min(angle + noise, 2 * math.pi))

    XX = TF.rotate(X, math.degrees(angle))

    return XX, torch.Tensor([noisy_sensor])
