#!/bin/bash
curdir=$(dirname "$0")
datasetroot="weizmann_horse_db"
xhidden=64
xsize=128
yhidden=256
depth=8
level=3
modelpath="log_20230305_2230/checkpoints/checkpoint_20000.pth.tar"


python train_main.py --dataset_root="$curdir/$datasetroot" --x_hidden_channels="$xhidden" --x_hidden_size="$xsize" --y_hidden_channels="$yhidden" --flow_depth="$depth" --num_levels="$level" --out_root="$outroot" --model_path="$curdir/$modelpath"