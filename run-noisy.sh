#!/bin/bash

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

    # CIFAR10 Scale Baseline
    "6c3ise2g" "t2l0pwfn" "dmry5bwz" "uzgmipvm" "wc0okcol"
    # CIFAR10 Scale Inverse
    "ptukvhie" "2yi37wzi" "ise4adf8" "3kr3nzgh" "xffbizcf"
    # CIFAR10 Rotation OFA
    "evb2gbfe" "lifpqbpe" "uz3d1y7i" "wdy9prc9" "4pafwqnr"
    # CIFAR10 Rotation OFA 2x
    "q218zovm" "pcge45np" "93mdak6l" "60vzvnko" "47t6t3ta"
    # CIFAR10 Rotation SCN D=3
    "dr2mx6gh" "m6plpu01" "q98azute" "kwfjs7kl" "jk4l92nv"
    # CIFAR10 Rotation SCN D=5
    "vctfowvy" "i1g43oz4" "jv937rv4" "7dxijvuv" "6750xv1s"
);

for id in "${id_array[@]}"; do
  python3 evaluate_additional.py $id --device cpu --num_workers 8
done;