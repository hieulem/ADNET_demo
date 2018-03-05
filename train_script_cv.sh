#!/bin/bash
root='/nfs/bigbox/hieule/p1000/'
trainingdir='CROPPED/p1000/training'
gpu=$1
name=p1000_3dataset
nc=3
CMD="python penguin_train.py --gpu_ids $gpu --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port 9998 --checkpoints ${root}checkpoints"
echo $CMD
eval $CMD
