import os
import torch

from sen12mscr_base import SEN12MSCR
from data.base_dataset_tensor import BaseDataset, get_params, get_transform

class SEN12MSCRAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # 这句为了计算FID-Epoch进行了修改，opt.dataset_phase
        # self.dir_AB = os.path.join(opt.dataroot, opt.dataset_phase)  # get the image directory
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        split = 'train' if opt.isTrain else 'test,val'
        self.sen12mscr = SEN12MSCR(opt.dataroot, split, season=opt.sen12mscr_season, rescale_method=opt.sen12mscr_rescale)
        self.length = len(self.sen12mscr)
        
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        sample = self.sen12mscr.__getitem__(index)
        A = sample['SAR']['S1'] # 0~1
        A = torch.Tensor(A)
        A_path = sample['SAR']['S1_path']
        
        B = sample['EO']['S2'] # 0~1
        B = torch.Tensor(B)
        B_path = sample['EO']['S2_path']

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (256, 256))
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.sen12mscr)