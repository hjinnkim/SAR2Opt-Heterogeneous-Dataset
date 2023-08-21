"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

# Edit configs with argparse
parser.add_argument("--use_argparse_configs", action="store_true", help='Edit config with argument parsing')
parser.add_argument('--data_root', type=str, default=None, help='path to images (should have subfolders trainA, trainB, testA, testB, or SEN12MS-CR)')
parser.add_argument('--name', type=str, default=None, help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--lr', type=float, default=None, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default=None, help='learning rate policy. [constant | step]')
parser.add_argument('--gan_w', type=float, default=None, help='weight of adversarial loss')
parser.add_argument('--recon_x_w', type=float, default=None, help='weight of image reconstruction loss')
parser.add_argument('', type=float, default=None, help='initial learning rate for adam')
parser.add_argument('--recon_s_w', type=float, default=None, help='weight of style reconstruction loss')
parser.add_argument('--recon_c_w', type=float, default=None, help='weight of content reconstruction loss')
parser.add_argument('--recon_x_cyc_w', type=float, default=None, help='weight of explicit style augmented cycle consistency loss')
parser.add_argument('--vgg_w', type=float, default=None, help='weight of domain-invariant perceptual loss')
parser.add_argument("--use_epoch_train", action="store_true")
parser.add_argument('--nepoch', type=int, default=100, help='train_epochs')
parser.add_argument('--batch_size', type=int, default=None, help='input batch size')
parser.add_argument('--num_workers', default=None, type=int, help='# threads for loading data')
parser.add_argument('--log_iter', type=int, default=1000, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--image_save_iter', type=int, default=4000, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--snapshop_save_epoch', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument("--use_sen12mscr", action="store_true")
parser.add_argument('--sen12mscr_season', type=str, default='all', help='chooses the season for SEN12MS-CR dataset. [all | spring | summer | fall | winter | season1,seaon2 | season1,season2,seaon3]') # You can choose multiple seasons via comma. e.g.) spring,summer
parser.add_argument('--s1_rescale_mtehod', type=str, default='default', help='chooses the rescale_method for SEN12MS-CR dataset. [default | norm | clip_1 | clip_2 | norm_1 | norm_2]')
parser.add_argument('--s2_rescale_mtehod', type=str, default='default', help='chooses the rescale_method for SEN12MS-CR dataset. [default | norm | clip_1 | clip_2 | norm_1 | norm_2]')
parser.add_argument('--s1_rgb_composite', type=str, default='mean', help='chooses the rescale_method for SEN12MS-CR dataset. [mean]')
parser.add_argument('--new_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--use_hsv_aug', action='store_true', help='EO random color jittering')
parser.add_argument('--use_gray_aug', action='store_true', help='EO random grayscaling')
parser.add_argument('--use_gaussian_blur', action='store_true', help='EO random gaussian blur')
parser.add_argument('--kernel_size', type=int, default=5, help='the size of guassisn blur')

opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Set config with argparse
if opts.use_argparse_configs:
    if not opts.data_root:
        config['data_root'] = opts.data_root
    if not opts.name:
        config['name'] = opts.name
    if not opts.lr:
        config['lr'] = opts.lr
    if not opts.lr_policy:
        config['lr_policy'] = opts.lr_policy
    if not opts.gan_w:
        config['gan_w'] = opts.gan_w
    if not opts.recon_x_w:
        config['recon_x_w'] = opts.recon_x_w
    if not opts.recon_s_w:
        config['recon_s_w'] = opts.recon_s_w
    if not opts.recon_c_w:
        config['recon_c_w'] = opts.recon_c_w
    if not opts.recon_x_cyc_w:
        config['recon_x_cyc_w'] = opts.recon_x_cyc_w
    if not opts.vgg_w:
        config['vgg_w'] = opts.vgg_w
    if opts.use_epoch_train:
        config['use_epoch_train'] = True
    else:
        config['use_epoch_train'] = False
    if opts.nepoch:
        config['train_epochs'] = opts.nepoch     
    if not opts.batch_size:
        config['batch_size'] = opts.batch_size
    if not opts.num_workers:
        config['num_workers'] = opts.num_workers
    if not opts.log_iter:
        config['log_iter'] = opts.log_iter
    if not opts.image_save_iter:
        config['image_save_iter'] = opts.image_save_iter
    if opts.use_sen12mscr:
        config['dataset'] = 'SEN12MS-CR'
    if opts.new_size:
        config['new_size'] = opts.new_size
    if opts.crop_size:
        config['crop_image_height'] = opts.crop_size
        config['crop_image_width'] = opts.crop_size
    if opts.use_hsv_aug:
        config['use_hsv_aug'] = True
    else:
        config['use_hsv_aug'] = False
    if opts.use_gray_aug:
        config['use_gray_aug'] = True
    else:
        config['use_gray_aug'] = False
    if opts.use_gaussian_blur:
        config['use_gaussian_blur'] = True
    else:
        config['use_gaussian_blur'] = False
    if opts.kernel_size:
        config['kernel_size'] = opts.kernel_size
        
if opts.use_sen12mscr:
    from sen12mscr.utils.utilsMUNIT import get_all_data_loaders
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(opts)
    if opts.use_epoch_train:
        loader_length = len(train_loader_a)
        max_iter = opts.nepoch * loader_length * opts.batch_size
        config['max_iter'] = max_iter
        step_size = max_iter // 10
        config['step_size'] = step_size
        if not opts.snapshop_save_epoch:
            config['snapshot_save_iter'] = opts.snapshop_save_epoch * loader_length
else:
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()

train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
if not opts.name:
    model_name = opts.name

train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# copy config file to output folder
if not opts.use_argparse_configs:
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) 
else:
    import yaml
    with open(os.path.join(output_directory, 'config.yaml'), 'w') as stream:
        yaml.dump(config, stream)
    


# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations))
            # HTML
            write_html(output_directory + "/index.html", iterations, config['image_save_iter'], 'images')

        if (iterations) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += opts.batch_size
        if iterations >= max_iter:
            sys.exit('Finish training')

