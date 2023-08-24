#!/bin/bash
#SBATCH -J nicegan_EO2SAR #Job이름
#SBATCH -p gpu_2080ti #gpu 할당
#SBATCH -N 1 #노드1개 (고정)
#SBATCH -n 4 #cpu4개(고정)
#SBATCH -o %x.o%j #-o = output, x = GAN(Job이름), j = JobID
#SBATCH -e %x.e%j #-e = error
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:1 #GPU 개수
#SBATCH --nodelist=gpu2080-04

# A : SAR, B : EO

# dataset           : dataset directory
# name              : saved directory name
# phase             : train / test
# light             : True / False (light / dense model)
# iteration         : training iterations
# use_epoch_train   : store_true (training w.r.t epochs)
# epoch             : training epochs (paper default setting / only with use_epoch_train)
# batch_size        : #TODO (paper default setting = 1)
# print_freq        : 1000 (print training results on result_dir (iter))
# save_freq         : 10000 (save model weights on result_dir (iter))
# save_epoch_freq   : 5 (save model weights on result dir / only with uie_epoch_train)
# decay_flag        : True (learning rate decay = linear)
# lr                : 0.001 (paper default setting = 0.001)
# adv_weight        : 1 (default setting)
# cycle_weight      : 10 (default setting)
# recon weight      : 10 (default setting)
# img_size          : 256 (image crop size)
# load_size         : 286 (image resize size)
# result_dir        : output directory
# use_sen12mscr     : store_true (use SEN12MS-CR dataset)
# use_hsv_aug       : store_true (EO Random Color Jitter) #TODO
# use_gray_aug      : store_true (EO Random Grayscale) #TODO
# use_gaussian_blur : store_true (EO Random Gaussian Blur) #TODO
# kernel_size       : 3 / 5 / 7 (EO Random Gaussian Blur kernel size) 
# benchmark_flag    : False (Do not benchmarking)

python NICE-GAN/main.py \
    --data_root /nas2/lait/5000_Dataset/Image/SEN12MSCR \
    --phase train \
    --light False \
    --iteration 0 \
    --use_epoch_train \
    --epoch 100 \
    --batch_size 2 \
    --print_freq 1000 \
    --save_freq 10000 \
    --save_epoch_freq 5 \
    --decay_flag True \
    --lr 0.001 \
    --adv_weight 1 \
    --cycle_weight 10 \
    --recon_weight 10 \
    --img_size 256 \
    --load_size 286 \
    --result_dir /home/haneollee/myworkspace_hj/SAR2Opt-Heterogeneous-Dataset/results \
    --use_sen12mscr \
    --sen12mscr_season spring \
    --s1_rescale_method norm \
    --s2_rescale_method norm \
    --s1_rgb_composite mean \
    --benchmark_flag False \
    # --use_hsv_aug \
    # --use_gray_aug \
    # --use_gaussian_blur \
    # --kernel_size 5 \
