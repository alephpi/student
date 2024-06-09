#!/bin/bash
curdir=$(dirname "$0")
datasetroot="logo_gui_db/embed"
logroot=""
xhidden=64
xsize=128
yhidden=256
depth=8
level=3
epochs=1001
batchsize=2
checkpointgap=1000
loggap=1
savegap=10000
infergap=10000
lr=0.0002
grad_clip=0.25
grad_norm=0
regularizer=0

python train_main.py --dataset_root="$curdir/$datasetroot" --log_root="$logroot" --x_hidden_channels="$xhidden" --x_hidden_size="$xsize" --y_hidden_channels="$yhidden" --flow_depth="$depth" --num_levels="$level" --num_epochs="$epochs" --batch_size="$batchsize" --checkpoints_gap="$checkpointgap" --nll_gap="$loggap" --inference_gap="$infergap" --lr="$lr" --max_grad_clip="$grad_clip" --max_grad_norm="$grad_norm" --save_gap="$savegap"  --regularizer="$regularizer"
