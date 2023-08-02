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
    --dataset SEN12MS-CR \
    --name nicegan_spring\
    --phase train \
    --light False \
    --iteration 0 \
    --batch_size \
    --print_freq 100 \
    --save_freq 10000 \
    --decay_flag True \
    --lr 0.001 \
    --img_size 256 \
    --load_size 256 \
    --result_dir \
    --benchmark_flag False \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --save_epoch_freq 5 \
    --sen12mscr \
    --sen12mscr_season spring \
    --sen12mscr_rescale default \
    --use_epoch_train

    # --no_flip \

    # For resuming
    # --resume True \
    # --checkpoint \

