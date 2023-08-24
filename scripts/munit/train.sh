#!/bin/bash
#SBATCH -J munit_EO2SAR #Job이름
#SBATCH -p gpu_2080ti #gpu 할당
#SBATCH -N 1 #노드1개 (고정)
#SBATCH -n 4 #cpu4개(고정)
#SBATCH -o %x.o%j #-o = output, x = GAN(Job이름), j = JobID
#SBATCH -e %x.e%j #-e = error
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1 #GPU 개수
#SBATCH --nodelist=gpu2080-04

# A : SAR, B : EO
# config            : MUNIT configuration file
# output_path       : output directory
# trainer           : MUNIT | UNIT (default : MUNIT)
# use_argparse_configs : store_true = edit configs with argparse
# data_root         : SEN12MS-CR dataset root directory (If not specified, data_root is set by config)
# name              : saved directory name
# lr                : 0.001 (paper default setting = 0.001 )
# lr_policy         : step (default for MUNIT)
# gan_w             : 1 (default setting)
# recon_x_w         : 10 (default setting)
# recon_s_w         : 1 (default setting)
# recon_c_w         : 1 (default setting)
# recon_x_cyc_w     : 0 (default setting)
# vgg_w             : 0 (default setting)
# use_epoch_train   : store_true (train w.r.t epochs)
# nepoch            : training epochs (paper default setting / only with use_epoch_train)
# batch_size        : #TODO (paper default setting = 1)
# num_workers       : threads for loading data (MUNIT default = 8)
# log_iter          : 1000 (print training results on logs (iter))
# image_save_iter   : 4000 (save images on logs (iter))
# snapshop_save_epoch : 5 (save model weights on logs (iter))
# use_sen12mscr     : store_true (use SEN12MS-CR dataset)
# new_size          : 286 (resize & crop <- resize)
# crop_size         : 256 (resize & crop <- crop : original data size)
# use_hsv_aug       : store_true (EO Random Color Jitter) #TODO
# use_gray_aug      : store_true (EO Random Grayscale) #TODO
# use_gaussian_blur : store_true (EO Random Gaussian Blur) #TODO
# kernel_size       : 3 / 5 / 7 (EO Random Gaussian Blur kernel size) #TODO
# sen12mscr_season  : all / spring / summer / fall / winter / (multi seasons in form : season1,season2,season3 / ex) spring,summer)
# s1_rescale_mtehod : default / norm / clip_1 / clip_2 / norm_1 / norm_2 #TODO
# s2_rescale_mtehod : default / norm / clip_1 / clip_2 / norm_1 / norm_2 #TODO
# s1_rgb_composite  : mean


python MUNIT/train.py \
    --config MUNIT/configs/opt2sar_sen12mscr.yaml \
    --output_path /home/haneollee/myworkspace_hj/SAR2Opt-Heterogeneous-Dataset/results \
    --trainer MUNIT \
    --use_argparse_configs \
    --data_root /nas2/lait/5000_Dataset/Image/SEN12MSCR \
    --name munit_spring \
    --lr 0.001 \
    --lr_policy step \
    --gan_w 1 \
    --recon_x_w 10 \
    --recon_s_w 1 \
    --recon_c_w 1 \
    --recon_x_cyc_w 0 \
    --vgg_w 0 \
    --use_epoch_train \
    --nepoch 100 \
    --batch_size 4 \
    --num_workers 8 \
    --log_iter 1000 \
    --image_save_iter 4000 \
    --snapshop_save_epoch 5 \
    --use_sen12mscr \
    --new_size 286 \
    --crop_size 256 \
    --sen12mscr_season spring \
    --s1_rescale_method clip_1 \
    --s2_rescale_method clip_1 \
    --s1_rgb_composite mean \
    --use_hsv_aug \
    --use_gray_aug \
    --use_gaussian_blur \
    --kernel_size 5 \
