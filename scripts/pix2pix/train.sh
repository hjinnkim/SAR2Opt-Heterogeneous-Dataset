#!/bin/bash
#SBATCH -J pix2pix_EO2SAR #Job이름
#SBATCH -p gpu_2080ti #gpu 할당
#SBATCH -N 1 #노드1개 (고정)
#SBATCH -n 4 #cpu4개(고정)
#SBATCH -o %x.o%j #-o = output, x = GAN(Job이름), j = JobID
#SBATCH -e %x.e%j #-e = error
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1 #GPU 개수
#SBATCH --nodelist=gpu2080-04

# A : SAR, B : EO

python train.py \
    --dataroot  \
    --name pix2pix_spring \
    --model pix2pix \
    --netG unet_256 \
    --direction BtoA \
    --dataset_mode sen12mscr_aligned \
    --sen12mscr_season spring \
    --sen12mscr_rescale default \
    --load_size 256 \
    --crop_size 256 \
    --batch_size \
    --norm instance \
    --pool_size 0 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --checkpoints_dir \
    --lr 0.001 \
    --save_epoch_freq 5 \
    --print_freq 100 \
    --display_id 1 \
    --display_port 8097 \


