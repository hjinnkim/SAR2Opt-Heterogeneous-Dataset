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
    --dataroot \
    --use_parse_configs \
    --use_epoch_train \
    --name munit_spring \
    --trainer MUNIT \
    --config configs/munit_spring.yaml \
    --sen12mscr \
    --sen12mscr_season spring \
    --sen12mscr_rescale default \
    --load_size 256 \
    --crop_size 256 \
    --batch_size \
    --norm none \
    --n_epochs 100 \
    --output_path \
    --lr 0.001 \
    --save_epoch_freq 5 \
    --print_freq 100 \
    # --resume