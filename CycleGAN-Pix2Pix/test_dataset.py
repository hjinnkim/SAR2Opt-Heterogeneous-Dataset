
from options.train_options import TrainOptions
from data.aligned_Pix2Pix_dataset import AlignedPix2PixDataset


opt = TrainOptions().parse()

dataset = AlignedPix2PixDataset(opt)

for i, sample in enumerate(dataset):
    print(i)