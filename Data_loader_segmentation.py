import os
from torch.utils import data
# import nibabel as nib
# import matplotlib.pyplot as plt
import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
# import cv2

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_sub_list(train_img_dir):
    image_list = []
    image_files = glob.glob(os.path.join(train_img_dir,"*.png"))
    image_files.sort()
    for name in image_files:
        file_name = os.path.basename(name)[:]
        name = file_name.split('.nii.gz')
        image_list.append(os.path.basename(name)[:])

    return image_list, image_files

def preprocess(image, flip=False, scale=None, crop=None):
  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      # mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
  if scale:
    w, h = image.size
    rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    # print(new_size)
    image = image.resize(new_size, Image.ANTIALIAS)
    # mask = mask.resize(new_size, Image.NEAREST)

  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

  image = data_transforms(image)
  # mask = torch.LongTensor(np.array(mask).astype(np.int64))

  if crop:
    h, w = image.shape[1], image.shape[2]
    pad_tb = max(0, crop[0] - h)
    pad_lr = max(0, crop[1] - w)
    image = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image)
    # mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

    h, w = image.shape[1], image.shape[2]
    i = random.randint(0, h - crop[0])
    j = random.randint(0, w - crop[1])
    image = image[:, i:i + crop[0], j:j + crop[1]]
    # mask = mask[i:i + crop[0], j:j + crop[1]]

  return image


class Data_loader(data.Dataset):
    def __init__(self, pv_label, pv_prior, pv_img, nc_label, nc_prior, nc_img, train=True, transform=None, target_transform=None, download=False, crop_size=None, dataset='multiorgan'):
        img_list = []
        label_list = []
        self.nc_label = nc_label
        self.nc_prior = nc_prior
        self.nc_img = nc_img

        self.pv_label = pv_label
        self.pv_prior = pv_prior
        self.pv_img = pv_img

        pv_image = sorted(self.pv_img)
        pv_label = sorted(self.pv_label)
        pv_morph_prior = sorted(self.pv_prior)
        nc_image = sorted(self.nc_img)
        nc_label = sorted(self.nc_label)
        nc_morph_prior = sorted(self.nc_prior)
        
        # for i in range(len(pv_image)):
        #     label_list.append(0)
        # for i in range(len(nc_image)):
        #     label_list.append(1)
        # for art in art_image:
        #     img_list.append(art)
        #     label_list.append(2)
        
        self.image = pv_image + nc_image
        self.prior = pv_morph_prior + nc_morph_prior
        self.label = pv_label + nc_label
        
        self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.3, 0.7)),
        transforms.RandomAffine((-30,+30)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def __getitem__(self, index):
        # print(self.image)
        img_name = self.image[index].split('/')[-1]
        _img = Image.open(self.image[index]).convert('RGB')
        _prior = Image.open(self.prior[index]).convert('RGB')
        p = torch.tensor(np.array(_prior)).permute(2, 0, 1)
        # # print(com_input.shape)
        # channel_input = Image.fromarray(np.uint8(com_input))
        input_img = self.test_transform(_img)
        f_img = torch.cat([input_img, p], dim=0)
        # print(input_img.size())
        # com_input = np.zeros((img.shape[0], img.shape[1], 6))
        # com_input[:, :, :3] = img
        # com_input[:, :, 3:] = prior
        # com_input = com_input.astype('float32')
        # # print(com_input.shape)
        # channel_input = Image.fromarray(np.uint8(com_input))
        # input_img = self.test_transform(com_input)
        # print(channel_input.size)
        # _img = _img.resize((int(256), int(256)))
        # _img_1 = self.train_transform(_img) 
        # _img_2 = self.train_transform(_img)  
        # print(x_1.shape)
        
        # print(self.image[index])
        label = Image.open(self.label[index]).convert('L')
        label_np = np.array(label)
        # print(np.unique(label_np))
        _label = torch.LongTensor(label_np.astype(np.int64))
        l_vol = np.ones((2, 128, 128))
        for i in range(2):
            seg_one = label_np == i
            l_vol[i,:,:] = seg_one[0:l_vol.shape[1],0:l_vol.shape[2]]
            l_vol[0,:,:] = l_vol[0,:,:] - l_vol[i,:,:]
        l_vol = l_vol.astype('float32')
        
        

        return f_img, _label, l_vol, img_name

    def __len__(self):
        return len(self.image)


