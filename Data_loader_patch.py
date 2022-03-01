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
    def __init__(self, pv_prior, pv_img, nc_prior, nc_img, train=True, transform=None, target_transform=None, download=False, crop_size=None, dataset='multiorgan'):
        pv_organ = {'spleen' : 0, 'rk' : 1, 'lk' : 2, 'gall' : 3, 'eso' : 4, 'liver' : 5, \
            'stomach' : 6, 'aorta' : 7, 'IVC' : 8, 'PSV' : 9, 'pancreas' : 10, 'rad' : 11}
        nc_organ = {'spleen' : 12, 'rk' : 13, 'lk' : 14, 'gall' : 15, 'eso' : 16, 'liver' : 17, \
            'stomach' : 18, 'aorta' : 19, 'IVC' : 20, 'PSV' : 21, 'pancreas' : 22, 'rad' : 23}
        # nc_organ = {'spleen' : 0, 'rk' : 1, 'lk' : 2, 'gall' : 3, 'eso' : 4, 'liver' : 5, \
        #     'stomach' : 6, 'aorta' : 7, 'IVC' : 8, 'PSV' : 9, 'pancreas' : 10, 'rad' : 11}

        img_list = []
        nc_label_list = []
        pv_label_list = []
        # self.nc_label = nc_label
        self.nc_prior = nc_prior
        self.nc_img = nc_img

        # self.pv_label = pv_label
        self.pv_prior = pv_prior
        self.pv_img = pv_img

        # num_pv = int(0.1 * len(self.pv_img))
        # num_nc = int(0.1 * len(self.nc_img))
        pv_image = sorted(self.pv_img)
        pv_morph_prior = sorted(self.pv_prior)
        nc_image = sorted(self.nc_img)
        nc_morph_prior = sorted(self.nc_prior)
        
        for i in range(len(pv_image)):
            organ = pv_image[i].split('.png_')[1].split('_')[0]
            organ_label = pv_organ[organ]
            pv_label_list.append(organ_label)
        for i in range(len(nc_image)):
            organ = nc_image[i].split('.png_')[1].split('_')[0]
            organ_label = nc_organ[organ]
            nc_label_list.append(organ_label)
        # for art in art_image:
        #     img_list.append(art)
        #     label_list.append(2)
        
        self.image = pv_image + nc_image
        self.prior = pv_morph_prior + nc_morph_prior
        self.label = pv_label_list + nc_label_list
        
        self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=128, scale=(0.3, 0.7)),
        transforms.RandomAffine((-30,+30)),
        transforms.RandomHorizontalFlip()])
        # transforms.ToTensor()])
        
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.test_transform = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        # print(self.image)
        img_name = self.image[index].split('/')[-1]
        _img = Image.open(self.image[index]).convert('RGB')
        # _img_np = np.array(_img)
        # _img_np = (_img_np - _img_np.min()) / (_img_np.max() - _img_np.min()).astype('uint8')
        # _img_norm = Image.fromarray(_img_np)
        _prior = Image.open(self.prior[index]).convert('RGB')
        s_1 = np.random.randint(2147483647)
        random.seed(s_1)
        torch.manual_seed(s_1)
        img_transform_1 = self.train_transform(_img) 
        _img_1 = self.norm_transform(img_transform_1)
        # (print(np.unique(_img_1.data.numpy())))
        random.seed(s_1)
        torch.manual_seed(s_1)
        prior_transform_1 = self.train_transform(_prior)
        _p_1 = torch.tensor(np.array(prior_transform_1)).permute(2, 0, 1)
        # (print(np.unique(_p_1.data.numpy())))
        
        s_2 = np.random.randint(1987687402)
        random.seed(s_2)
        torch.manual_seed(s_2)
        img_transform_2 = self.train_transform(_img)
        _img_2 = self.norm_transform(img_transform_2)
        random.seed(s_2)
        torch.manual_seed(s_2)
        prior_transform_2 = self.train_transform(_prior)
        _p_2 = torch.tensor(np.array(prior_transform_2)).permute(2, 0, 1)
        # eps = 1e-6

        # _img_1 = np.zeros((128, 128, 6))
        # img_np_1 = np.array(img_transform_1)
        # print(np.unique(img_np_1))
        # img_np_1 = (img_np_1 - img_np_1.min() + eps) / (img_np_1.max() - img_np_1.min() + eps)
        # print(np.unique(img_np_1))
        # prior_np_1 = np.array(prior_transform_1)
        # print(prior_np_1.shape)
        # _img_1[:, :, :3] = img_np_1
        # _img_1[:, :, 3:] = prior_np_1
        # _img_1 = _img_1.astype('float32')

        # _img_2 = np.zeros((128, 128, 6))
        # img_np_2 = np.array(img_transform_2)
        # img_np_2 = (img_np_2 - img_np_2.min() + eps) / (img_np_2.max() - img_np_2.min() + eps)
        # print(np.unique(img_np_2))
        # prior_np_2 = np.array(prior_transform_2)
        # _img_2[:, :, :3] = img_np_2
        # _img_2[:, :, 3:] = prior_np_2
        # _img_2 = _img_2.astype('float32')

        # _img_1 = self.test_transform(_img_1)
        # _img_2 = self.test_transform(_img_2)
        # print(x_1.shape)
        
        # print(self.image[index])
        label = self.label[index]
        # x_2 = np.zeros((24))
        # x_2[label] = 1
        # x_2 = x_2.astype('float32')
        x_2 = np.reshape(label,(1,))
        x_2 = x_2.squeeze()
        
        

        return [_img_1, _img_2], [_p_1, _p_2], x_2, img_name

    def __len__(self):
        return len(self.image)


