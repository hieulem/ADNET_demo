#!/bin/bash
module load cuda80/toolkit/8.0.44
module load shared
module load anaconda/3
source activate pytorch
root='/gpfs/projects/LynchGroup/CROZtrain/'
trainingdir='CROPPED/p1000/training'
name=single_p1000_2
CMD="python /gpfs/home/hle/code/ADNET_demo/penguin_train.py --name $name --root $root --dataroot ${root}${trainingdir} --dataset_mode tif --fineSize 256 --display_port 9998 --checkpoints ${root}checkpoints"
echo $CMD
eval $CMD>/gpfs/home/hle/code/ADNET_demo/log/$name 
