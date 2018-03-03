#!/bin/bash
module load torque/6.0.2
qsub -lwalltime=40:99:99 -q gpu-long train_script.sh -o /gpfs/home/hle/code/ADNET_demo/trainning.txt -e /gpfs/home/hle/code/ADNET_demo/errr.txt 
