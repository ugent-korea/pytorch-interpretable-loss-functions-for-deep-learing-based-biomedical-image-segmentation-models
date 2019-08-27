#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 19:14:26 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""

from PIL import Image
import glob
import numpy as np

# Train_images
from_folder_name = 'raw_images'
to_folder_name = 'clean_images'
train_im_list = glob.glob('data/'+from_folder_name+'/*')

for image_path in train_im_list:
    image_name = image_path[image_path.rindex('/')+1:-4]
    image_name = image_name.replace('-', '_')
    print(np.array(Image.open(image_path)).shape)
    im = np.array(Image.open(image_path)).transpose(2, 0, 1)
    d, h, w = im.shape
    print(im.shape, im[0, 0].shape)

    im_first_part = im[:, :, :1072].transpose(1, 2, 0)
    f = Image.fromarray(im_first_part)
    # f.show()
    f.thumbnail((w*0.4, h*0.4), Image.ANTIALIAS)
    f.save('data/'+to_folder_name+'/'+image_name+'_1.png')

"""
from_folder_name = 'raw_masks'
to_folder_name = 'clean_masks'
train_im_list = glob.glob('../data/'+from_folder_name+'/*')

for image_path in train_im_list:
    image_name = image_path[image_path.rindex('/')+1:-14]
    image_name = image_name.replace('-', '_')
    im = np.array(Image.open(image_path))
    im_first_part = im[:, :1072]
    h, w = im.shape

    f = Image.fromarray(im_first_part)
    f.thumbnail((w*0.4, h*0.4), Image.ANTIALIAS)

    f.save('../data/'+to_folder_name+'/'+image_name+'.png')
"""
