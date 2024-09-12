#!/bin/bash

# pkg update && pkg upgrade -y
# termux-setup-storage
# pkg install proot-distro
# proot-distro install ubuntu
# proot-distro login ubuntu
# apt update && apt upgrade -y && apt install python3 python3-pip -y
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages
# pip3 install wandb tqdm --break-system-packages
# wandb login
# cp -r /storage/emulated/0/Download/sensor_guided_adaptation .
# cd sensor_guided_adaptation/
# chmod +x run-benchmark.sh && ./run-benchmark.sh

id_array=(
    # CIFAR10 Rotation Baseline
    "delojfvs" "pjxyww6u" "r4cbti93" "smi8lx67" "iovapdx3"
    # CIFAR10 Rotation Inverse
    "l2i4o8yg" "7f53abst" "s5nhlc88" "1oj76wi2" "s9fw6g57"
    # CIFAR10 Rotation OFA
    "dqjajoin" "60s6jrg0" "vqpvhznn" "jzlpdgq7" "r1yv30ne"
    # CIFAR10 Rotation OFA 2x
    "bm4lm2gi" "mpy56la0" "nx5hglp4" "u3nbyl8u" "ydxobb3c"
    # CIFAR10 Rotation SCN D=3
    "jgreesri" "8fhyrrp4" "i6lyjim8" "h96hxs2m" "npkdshw8"
    # CIFAR10 Rotation SCN D=5
    "od36byym" "rlw9l3x1" "r94azafc" "czz5nv4x" "htia3vx1"
);

for id in "${id_array[@]}"; do
    python3 evaluate_additional.py $id runtime --device cpu --num_workers 1 --batch_size 1
#    python3 evaluate_additional.py $id runtime --device cpu --num_workers 6 --batch_size 128
#    python3 evaluate_additional.py $id runtime --device cpu --num_workers 6 --batch_size 256
done;