#!/bin/bash
#SBATCH -J cut_EO2SAR #Job이름
#SBATCH -p gpu_2080ti #gpu 할당
#SBATCH -N 1 #노드1개 (고정)
#SBATCH -n 4 #cpu4개(고정)
#SBATCH -o %x.o%j #-o = output, x = GAN(Job이름), j = JobID
#SBATCH -e %x.e%j #-e = error
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1 #GPU 개수
#SBATCH --nodelist=gpu2080-04

# A : SAR, B : EO

# dataroot          : SEN12MS-CR dataset root directory
# name              : saved directory name
# model             : cut
# CUT_mode          : CUT | FastCUT
# netG              : resnet_9blocks (default architecture)
# direction         : BtoA (EO to SAR)
# sen12mscr_season  : all / spring / summer / fall / winter / (multi seasons in form : season1,season2,season3 / ex) spring,summer)
# s1_rescale_mtehod : default / norm / clip_1 / clip_2 / norm_1 / norm_2 #TODO
# s2_rescale_mtehod : default / norm / clip_1 / clip_2 / norm_1 / norm_2 #TODO
# s1_rgb_composite  : mean
# load_size         : 286 (resize & crop <- resize)
# crop_size         : 256 (resize & crop <- crop : original data size)
# batch_size        : #TODO (paper default setting = 1)
# pool_size         : 50 (default setting)
# n_epoch           : 50 (50 epochs + 50 weight decaying epochs = total 100 epochs)
# n_epochs_decay    : 50 (50 epochs + 50 weight decaying epochs = total 100 epochs)
# checkpoints_dir   : #TODO
# lr                : 0.001 (paper default setting = 0.001 )
# lr_policy         : linear (default for CUT)
# save_epoch_freq   : 5 (default setting)
# print_freq        : 1000 (print training results on console (iter))
# display_freq      : 4000 (print training results on screen (iter))
# update_html_freq  : 10000 (saving training results to html (iter))
# display_id        : 4 (visualizer id : cut = 4)
# display_port      : 8100 (visualizer port : cut=8100)
# use_hsv_aug       : store_true (EO Random Color Jitter) #TODO
# use_gray_aug      : store_true (EO Random Grayscale) #TODO
# use_gaussian_blur : store_true (EO Random Gaussian Blur) #TODO
# kernel_size       : 3 / 5 / 7 (EO Random Gaussian Blur kernel size) #TODO

python train.py --dataroot ./datasets/sar2opt --name sar2opt_CUT --CUT_mode CUT


python train.py \
    --dataroot  \
    --name cut_spring \
    --model cut \
    --CUT_mode CUT \
    --netG resnet_9blocks \
    --direction BtoA \
    --dataset_mode unaligned_datasetCycleGAN \
    --sen12mscr_season spring \
    --s1_rescale_mtehod default \
    --s2_rescale_mtehod default \
    --s1_rgb_composite mean \
    --load_size 286 \
    --crop_size 256 \
    --batch_size \
    --pool_size 50 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --checkpoints_dir \
    --lr 0.001 \
    --lr_policy linear \
    --save_epoch_freq 5 \
    --print_freq 1000 \
    --display_freq 4000 \
    --display_id 4 \
    --display_port 8100 \
    --update_html_freq 10000 \
    # --use_hsv_aug \
    # --use_gray_aug \
    # --use_gaussian_blur \
    # --kernel_size 5 \