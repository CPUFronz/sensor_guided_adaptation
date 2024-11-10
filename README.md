# Sensor-Guided Adaptive Machine Learning  on Resource-Constrained Devices

This is the code for the [IoT 2024](https://iot-conference.org/iot2024/) paper ["Sensor-Guided Adaptive Machine Learning  on Resource-Constrained Devices"](https://doi.org/10.1145/3703790.3703801).

## Setup

A Conda ```*.yml``` file is provided for recreating the development environment. Just run ```conda env create -f environment.yml``` to create the required development environment.

## How to Run

The files to run the experiments are:

- [baseline.py](baseline.py)
- [cifar10_rotation.py](cifar10_rotation.py)
- [cifar10_scale.py](cifar10_scale.py)
- [gtsrb_rotation.py](gtsrb_rotation.py)
- [gtsrb_scale.py](gtsrb_scale.py)
- [inverse.py](inverse.py)

Note, that inverse requires a trained baseline model to run, the wandb id of the trained baseline model has to be passed as argument.

The accuracy of the resulting models is automatically evaluated after training, to evaluate the robustness against noisy sensors run [evaluate_noisy.py](evaluate_noisy.py) with the wandb id of the model to evaluate. To evaluate the runtime, you need ```adb``` installed and the Android phone connected. Change the wandb ids in ```id_array``` in [run-noisy.sh](run-noisy.sh) and start this Bash script.

## Data

We use the [GTSRB](https://benchmark.ini.rub.de/gtsrb_about.html) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets, both are available in PyTorch and will be automatically downloaded.

## Citing

If you find our work useful, please cite it using the following BibTex entry:

```
@inproceedings{papst2024,
  title = {Sensor-Guided Adaptive Machine Learning  on Resource-Constrained Devices},
  author = {Papst, Franz and Kraus, Daniel and Rechberger, Martin and Saukh, Olga},
  booktitle = {14th International Conference on the Internet of Things (IoT 2024)},
  year = {2024},
  doi = {10.1145/3703790.3703801}
}
```