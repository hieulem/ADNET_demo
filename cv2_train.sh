#!/bin/bash
temp=$1
root='/nfs/bigbox/hieule/penguin_data/CROPPED/'${temp}'/PATCHES/64_386/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=$2
nc=3
bias=0.5
name="nc${nc}_${temp}_bias${bias}"
checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints/'
CMD="python penguin_train.py --biased_sampling $bias  --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port 9998 --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
