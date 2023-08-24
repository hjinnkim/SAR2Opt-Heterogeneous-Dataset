from NICE import NICE
import argparse
from utils import *
import time 


"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of NICE-GAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_root', type=str, default=None, help='path to images (root to SEN12MS-CR)')
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=True, help='[NICE-GAN full version / NICE-GAN light version]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=194800, help='The number of training iterations')
    parser.add_argument("--use_epoch_train", action="store_true")
    parser.add_argument('--epoch', type=int, default=0, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='The number of model save freq in epoch')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--recon_weight', type=int, default=10, help='Weight for Reconstruction')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=6, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=7, help='The number of discriminator layer')
    parser.add_argument('--load_size', type=int, default=286, help='The resizing image size')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='Image_translation_codes/NICE-GAN-pytorch/results', help='Directory name to save the results')
    parser.add_argument("--use_sen12mscr", action="store_true")
    parser.add_argument('--sen12mscr_season', type=str, default='all', help='chooses the season for SEN12MS-CR dataset. [all | spring | summer | fall | winter | season1,seaon2 | season1,season2,seaon3]') # You can choose multiple seasons via comma. e.g.) spring,summer
    parser.add_argument('--s1_rescale_method', type=str, default='default', help='chooses the rescale_method for SEN12MS-CR dataset. [default | norm | clip_1 | clip_2 | norm_1 | norm_2]')
    parser.add_argument('--s2_rescale_method', type=str, default='default', help='chooses the rescale_method for SEN12MS-CR dataset. [default | norm | clip_1 | clip_2 | norm_1 | norm_2]')
    parser.add_argument('--s1_rgb_composite', type=str, default='mean', help='chooses the rescale_method for SEN12MS-CR dataset. [mean]')
    parser.add_argument('--use_hsv_aug', action='store_true', help='EO random color jittering')
    parser.add_argument('--use_gray_aug', action='store_true', help='EO random grayscaling')
    parser.add_argument('--use_gaussian_blur', action='store_true', help='EO random gaussian blur')
    parser.add_argument('--kernel_size', type=int, default=5, help='the size of guassisn blur')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
    parser.add_argument('--n_downsampling', type=int, default=2, help='The number of downsampling')
    
    parser.add_argument('--dataset_phase', type=str, default='train')
    parser.add_argument('--checkpoint', type=str, default='')


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, os.path.basename(args.dataset), 'model'))
    check_folder(os.path.join(args.result_dir, os.path.basename(args.dataset), 'img'))
    check_folder(os.path.join(args.result_dir, os.path.basename(args.dataset), 'fakeA'))
    check_folder(os.path.join(args.result_dir, os.path.basename(args.dataset), 'fakeB'))

    # --epoch
    try:
        assert args.epoch >= 1 or args.iteration >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = NICE(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    start_time = time.time()
    
    main()

    end_time = time.time()
    cost_time = end_time - start_time
    print('cost time: ', cost_time)
