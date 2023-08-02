import os
import torch

from sen12mscr_base import SEN12MSCR
from base_dataset_tensor import BaseDataset, get_params, get_transform
import random


class SEN12MSCRDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, return_A=True):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # 这两句为了计算FID-Epoch进行了修改，opt.dataset_phase
        # self.dir_A = os.path.join(opt.dataroot, opt.dataset_phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.dataset_phase + 'B')  # create a path '/path/to/data/trainB'
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        split = 'train' if opt.isTrain else 'test,val'
        self.sen12mscr = SEN12MSCR(opt.dataroot, split, season=opt.sen12mscr_season, rescale_method=opt.sen12mscr_rescale)
        self.length = len(self.sen12mscr)

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        self.return_A = return_A

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.return_A:
            A_sample = self.sen12mscr.get_SAR(index)  # make sure index is within then range
            A = A_sample['SAR']['S1'] # 0~1
            A = torch.Tensor(A)
            # apply image transformation
            A = self.transform_A(A)
            return A
        else:
            B_sample = self.sen12mscr.get_EO(index)
            B = B_sample['EO']['S2'] # 0~1
            B = torch.Tensor(B)
            # apply image transformation
            B = self.transform_B(B)
            return B

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.length
