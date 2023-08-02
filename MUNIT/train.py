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
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--sen12mscr', action='store_true', help='whether dataset is SEN12MS-CR dataset') # A : SAR(S1/VV,VH), B : EO(S2/B4,B3,B2)
parser.add_argument('--sen12mscr_season', type=str, default='all', help='chooses the season for SEN12MS-CR dataset. [all | spring | summer | fall | winter | season1,seaon2 | season1,season2,seaon3]') # You can choose multiple seasons via comma. e.g.) spring,summer
parser.add_argument('--sen12mscr_rescale', type=str, default='default', help='chooses the rescale_method for SEN12MS-CR dataset. [default | gaussian_normalize]')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
parser.add_argument('--n_epochs', type=int, default=-1, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--save_epoch_freq', type=int, default=-1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--use_parse_configs", action="store_true")
parser.add_argument("--use_epoch_train", action="store_true")
parser.add_argument("--resume", action="store_true")


opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
if opts.use_parse_configs:
    config['use_parse_configs'] = opts.use_parse_configs
    config['batch_size'] = opts.batch_size
    config['data_root'] = opts.dataroot
    config['new_size'] = opts.load_size
    config['crop_image_height'] = opts.crop_size
    config['crop_image_width'] = opts.crop_size
    config['lr'] = opts.lr
    config['norm'] = opts.norm
    config['log_iter'] = opts.print_freq
    config['input_dim_a'] = opts.input_nc
    config['input_dim_b'] = opts.output_nc
if opts.sen12mscr:
    config['dataset'] = 'SEN12MS-CR'
    config['sen12mscr_season'] = opts.sen12mscr_season
    config['sen12mscr_rescalenorm'] = opts.sen12mscr_rescalenorm

display_size = config['display_size']

if opts.sen12mscr:
    opts.no_flip = False
    opts.preprocess = 'resize_and_crop'
    opts.direction = 'AtoB'
    opts.isTrain = True
    num_threads = 4
    serial_batches = False
    from sen12mscr_dataset import SEN12MSCRDataset
    dataset_A = SEN12MSCRDataset(opts, return_A=True)
    dataset_B = SEN12MSCRDataset(opts, return_A=False)
    train_loader_a = torch.utils.data.DataLoader(dataset=dataset_A, batch_size=config['batch_size'], shuffle=serial_batches, drop_last=True, num_workers=serial_batches)
    train_loader_b = torch.utils.data.DataLoader(dataset=dataset_B, batch_size=config['batch_size'], shuffle=serial_batches, drop_last=True, num_workers=serial_batches)
else:
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

max_iter = config['max_iter']
config['vgg_model_path'] = opts.output_path

if opts.use_parse_configs and opts.use_epoch_train:
    config['use_epoch_train'] = opts.use_epoch_train
    config['max_iter'] = max_iter = len(train_loader_a) * opts.n_epochs
    config['step_size'] = int(config['max_iter'] / config['step_size'] * max_iter)
    config['save_epoch_iter'] = len(train_loader_a) * opts.save_epoch_freq

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()

# Setup logger and output folders

# model_name = os.path.splitext(os.path.basename(opts.config))[0]
# train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
# output_directory = os.path.join(opts.output_path + "/outputs", model_name)
# checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

model_name = os.path.splitext(os.path.basename(opts.config))[0] if opts.name == '' else opts.name
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path, model_name, "logs"))
output_directory = os.path.join(opts.output_path, model_name, "outputs")
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
if opts.use_parse_configs:
    import yaml
    with open(os.path.join(output_directory, 'config.yaml'), 'w') as stream:
        yaml.dump(config, stream)
else:
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0

if opts.sen12mscr:
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
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
#             if (iterations + 1) % config['image_save_iter'] == 0:
#                 with torch.no_grad():
#                     test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
#                     train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
#                 write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
#                 write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
#                 # HTML
#                 write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
# 
#             if (iterations + 1) % config['image_display_iter'] == 0:
#                 with torch.no_grad():
#                     image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
#                 write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)
            if opts.use_epoch_train and (iterations + 1) % config['save_epoch_iter'] == 0:
                trainer.save(checkpoint_directory+'/epoch_train', iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')


else:
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
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

