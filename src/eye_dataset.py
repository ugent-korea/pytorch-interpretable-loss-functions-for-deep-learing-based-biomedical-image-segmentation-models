import numpy as np
import glob
import random
import torch
from PIL import Image
from pre_process import *
from torch.utils.data.dataset import Dataset
np.random.seed(99)
random.seed(99)
torch.manual_seed(99)
# from pre_process import *


class EyeDataset(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        print('Train size:', self.data_len)
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        # Find image
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/')+1:]
        # Read image
        im_as_im = Image.open(image_path)
        im_as_np = np.asarray(im_as_im)
        im_as_np = im_as_np.transpose(2, 0, 1)
        # Crop image
        im_height, im_width = im_as_np.shape[1], im_as_np.shape[2]
        hcrop_start = random.randint(0, im_height - self.out_size)
        wcrop_start = random.randint(10, (im_width - self.out_size))
        im_as_np = im_as_np[:,
                            hcrop_start:hcrop_start+self.out_size,
                            wcrop_start:wcrop_start+self.out_size]
        # Flip image

        flip_num = random.randint(0, 3)
        im_as_np = flip(im_as_np, flip_num)
        # Pad image
        pad_size = int((self.in_size - self.out_size)/2)
        im_as_np = np.asarray([np.pad(single_slice, pad_size, mode='edge')
                               for single_slice in im_as_np])
        """
        # Sanity check
        img1 = Image.fromarray(im_as_np.transpose(1, 2, 0))
        img1.show()
        """
        # Normalize image
        im_as_np = im_as_np/255
        # im_as_np = np.expand_dims(im_as_np, axis=0)  # add additional dimension
        im_as_tensor = torch.from_numpy(im_as_np).float()  # Convert numpy array to tensor

        # --- Mask --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        msk_as_np = np.asarray(msk_as_im)
        # Crop mask
        msk_as_np = msk_as_np[hcrop_start:hcrop_start+self.out_size,
                              wcrop_start:wcrop_start+self.out_size]
        msk_as_np.setflags(write=1)
        msk_as_np[msk_as_np > 20] = 255
        # Flip mask
        if flip_num in [0, 1]:
            msk_as_np = np.flip(msk_as_np, flip_num)
        if flip_num == 2:
            msk_as_np = np.flip(msk_as_np, 0)
            msk_as_np = np.flip(msk_as_np, 1)
        # Pad mask
        # msk_as_np = np.pad(msk_as_np, pad_size, mode='edge')
        """
        # Sanity check
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        """
        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        return (image_name, im_as_tensor, msk_as_tensor)

    def __len__(self):
        return self.data_len


class EyeDatasetVal(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        print('Test size:', self.data_len)
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        # Find image
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/')+1:]
        # Read image
        im_as_im = Image.open(image_path)
        im_as_np = np.asarray(im_as_im)
        im_as_np = im_as_np.transpose(2, 0, 1)
        # Crop image
        im_height, im_width = im_as_np.shape[1], im_as_np.shape[2]
        hcrop_start = int((im_height - self.out_size)/2)
        wcrop_start = int(10 + (im_width - 10 - self.out_size)/2)
        im_as_np = im_as_np[:,
                            hcrop_start:hcrop_start+self.out_size,
                            wcrop_start:wcrop_start+self.out_size]

        # Pad image
        pad_size = int((self.in_size - self.out_size)/2)
        im_as_np = np.asarray([np.pad(single_slice, pad_size, mode='edge')
                               for single_slice in im_as_np])
        """
        # Sanity check
        img1 = Image.fromarray(im_as_np.transpose(1, 2, 0))
        img1.show()
        """
        # Normalize image
        im_as_np = im_as_np/255
        # im_as_np = np.expand_dims(im_as_np, axis=0)  # add additional dimension
        im_as_tensor = torch.from_numpy(im_as_np).float()  # Convert numpy array to tensor

        # --- Mask --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        msk_as_np = np.asarray(msk_as_im)
        # Crop mask
        msk_as_np = msk_as_np[hcrop_start:hcrop_start+self.out_size,
                              wcrop_start:wcrop_start+self.out_size]
        msk_as_np.setflags(write=1)
        msk_as_np[msk_as_np > 20] = 255
        """
        # Sanity check
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        """
        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        return (image_name, im_as_tensor, msk_as_tensor)

    def __len__(self):
        return self.data_len


class EyeDatasetTest(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        # paths to all images and masks
        self.mask_path = mask_path
        self.image_list = glob.glob(str(image_path) + str("/*"))
        self.data_len = len(self.image_list)
        print('Dataset size:', self.data_len)
        self.in_size, self.out_size = in_size, out_size

    def __getitem__(self, index):
        # Find image
        image_path = self.image_list[index]
        image_name = image_path[image_path.rfind('/')+1:]
        # Read image
        im_as_im = Image.open(image_path)
        im_as_np = np.asarray(im_as_im)
        im_as_np = im_as_np.transpose(2, 0, 1)
        # Crop image
        im_as_np = im_as_np[:,
                            70:70+408,
                            10:-10]  # 428 x 428
        # Pad image
        pad_size = 82
        im_as_np = np.asarray([np.pad(single_slice, pad_size, mode='edge')
                               for single_slice in im_as_np])
        """
        # Sanity check
        img1 = Image.fromarray(im_as_np.transpose(1, 2, 0))
        img1.show()
        """

        # Normalize image
        im_as_np = im_as_np/255
        # im_as_np = np.expand_dims(im_as_np, axis=0)  # add additional dimension
        im_as_tensor = torch.from_numpy(im_as_np).float()  # Convert numpy array to tensor

        # --- Mask --- #
        # Read mask
        msk_as_im = Image.open(self.mask_path + '/' + image_name)
        msk_as_np = np.asarray(msk_as_im)
        # Crop mask
        msk_as_np = msk_as_np[70:70+388, 20:-20]
        msk_as_np.setflags(write=1)
        msk_as_np[msk_as_np > 20] = 255
        # Sanity check
        """
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        """

        # Normalize mask to only 0 and 1
        msk_as_np = msk_as_np/255
        # msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
        msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
        return (image_name, im_as_tensor, msk_as_tensor)

    def __len__(self):
        return self.data_len


"""
x = EyeDatasetTest('../data/ts_images', '../data/clean_masks')
a, b = x.__getitem__(6)
print(a.size())
"""

"""
x = EyeDataset('../data/clean_images', '../data/clean_masks')
a, b = x.__getitem__(1)
print(a.size())
"""
