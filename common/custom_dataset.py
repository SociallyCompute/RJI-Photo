import torch, pandas as pd, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import transform

import os
from torchvision import transforms, utils

class AVAImagesDataset(Dataset):
    def __init__(self, ratings_file, root_dir, transform=None):
        self.ava_frame = pd.read_csv(ratings_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ava_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(self.ava_frame.iloc[idx, 0]) + '.jpg')
        if not os.path.isfile(img_name):
            return None
        image = np.array(Image.open(img_name))
        ratings = np.array([self.ava_frame.iloc[idx, 2:11]])
        ratings = ratings.astype('float').reshape(-1, 9)
        sample = {'image': image, 'ratings': ratings}

        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image, ratings = sample['image'], sample['ratings']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        # ratings = ratings * [new_w / w, new_h / h]
        return {'image':img, 'ratings':ratings}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        image, ratings = sample['image'], sample['ratings']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        # ratings = ratings - [left, top]

        return {'image': image, 'ratings': ratings}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, ratings = sample['image'], sample['ratings']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'ratings': torch.from_numpy(ratings)}


# fig = plt.figure()
# count = 0

# for i in range(len(ava_dataset)):
#     sample = ava_dataset[i]
#     if not sample:
#         continue
#     print(i, sample['image'].shape, sample['ratings'].shape)
#     # ax = plt.subplot(1, 4, count+1)
#     # plt.tight_layout()
#     # ax.set_title('Sample #{}'.format(i))
#     # ax.axis('off')
#     # show_image(**sample)
#     if count == 3:
#         # plt.show()
#         break
#     count += 1

ava_dataset = AVAImagesDataset(ratings_file='/media/matt/New Volume/ava/AVA.txt', 
                               root_dir='/media/matt/New Volume/ava/ava-compressed/images/',
                               transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

dataloader = DataLoader(ava_dataset, batch_size=4, shuffle=True, num_workers=4)

def show_ratings_batch(sample_batched):
    """Show image with ratings for a batch of samples."""
    images_batch, ratings_batch = \
            sample_batched['image'], sample_batched['ratings']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(ratings_batch[i, :, 0].numpy() + i * im_size,
                    ratings_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['ratings'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_ratings_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break