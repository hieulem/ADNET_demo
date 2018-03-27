#!/bin/bash
temp=$1
root='/nfs/bigbox/hieule/penguin_data/CROPPED/'${temp}'/PATCHES/64_386/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=$2
name="train_on_$temp"
nc=4
checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints4c/'
CMD="python penguin_train.py --gpu_ids $gpu --model single_unet_4c --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port 9998 --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
