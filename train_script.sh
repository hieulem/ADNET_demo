#!/bin/bash
module load cuda80/toolkit/8.0.44
module load shared
module load anaconda/3
source activate pytorch
#root='/gpfs/projects/LynchGroup/CROZtrain/'
root='/nfs/bigbox/hieule/p1000/
trainingdir='CROPPED/p1000/training'
name=single_p1000_1c
nc=3
CMD="python /gpfs/home/hle/code/ADNET_demo/penguin_train.py --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port 9998 --checkpoints ${root}checkpoints"
echo $CMD
eval $CMD>/gpfs/home/hle/code/ADNET_demo/log/$name 
