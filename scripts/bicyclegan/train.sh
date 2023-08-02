#!/bin/sh
#SBATCH -J uirp ## Job Name
#SBATCH -p gpu_2080ti ## Partition name
#SBATCH -N 1 ## node count 총 필요한 컴퓨팅 노드 수
#SBATCH -n 4 ## total number of tasks across all nodes 총 필요한 프로세스 수
#SBATCH -o %x.o%j ## filename of stdout, stdout 파일 명(.o)
#SBATCH -e %x.e%j ## filename of stderr, stderr 파일 명(.e)
#SBATCH --time 48:00:00 ##y 최대 작업 시간(Wall Time Clock Limit)
#SBATCH --gres=gpu:1 ## number of GPU(s) per node
#SBATCH --nodelist=gpu2080-03

# A : SAR, B : EO

python train.py \
    --dataroot \
    --name bicyclegan_spring \
    --model bycycle_gan \
    --direction BtoA \
    --dataset_mode sen12mscr_aligned \
    --sen12mscr_season spring \
    --sen12mscr_rescale default \
    --load_size 256 \
    --crop_size 256 \
    --batch_size \
    --norm instance \
    --n_epochs 50 \
    --n_epochs_decay 50 \
    --use_dropout \
    --checkpoints_dir \
    --lr 0.001 \
    --save_epoch_freq 5 \
    --display_id 3 \
    --display_port 8099 \
