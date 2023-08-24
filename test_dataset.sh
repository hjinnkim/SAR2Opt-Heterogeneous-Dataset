python CycleGAN-Pix2Pix/test_dataset.py \
    --dataroot /nas2/lait/5000_Dataset/Image/SEN12MSCR\
    --name pix2pix_spring \
    --model pix2pix \
    --netG unet_256 \
    --direction BtoA \
    --dataset_mode aligned_Pix2Pix \
    --sen12mscr_season spring \
    --s1_rescale_method norm \
    --s2_rescale_method norm \
    --s1_rgb_composite mean \
    --load_size 286 \
    --crop_size 256 \
    --batch_size 1 \
    --norm instance \
    --pool_size 0 \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --checkpoints_dir /home/haneollee/myworkspace_hj/SAR2Opt-Heterogeneous-Dataset\
    --lr 0.001 \
    --lr_policy linear \
    --lambda_L1 100.0 \
    --save_epoch_freq 5 \
    --print_freq 1000 \
    --display_freq 4000 \
    --display_id 1 \
    --display_port 8097 \
    --update_html_freq 10000 \
    # --use_hsv_aug \
    # --use_gray_aug \
    # --use_gaussian_blur \
    # --kernel_size 5 \