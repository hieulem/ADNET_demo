#!/bin/bash
root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=$1
name=$2
nc=3
CMD="python penguin_train.py --gpu_ids $gpu --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port 9999 --checkpoints ${root}checkpoints"
echo $CMD
eval $CMD
