# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os
import pickle
import random

import cv2
import numpy as np
import torch

from torch.utils import data as data
from torch.utils.data import Dataset


def paired_raw_random_crop(img_gt, img_lq, crop_size):
    h_lq, w_lq = img_lq.shape[-2:]
    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - crop_size)
    left = random.randint(0, w_lq - crop_size)
    # crop lq patch
    if img_lq.dim() == 4:
        img_lq = img_lq[:, :, top:top + crop_size, left:left + crop_size]
        img_gt = img_gt[:, :, top:top + crop_size, left:left + crop_size]
    elif img_lq.dim() == 3:
        img_lq = img_lq[:, top:top + crop_size, left:left + crop_size]
        img_gt = img_gt[:, top:top + crop_size, left:left + crop_size]
    else:
        raise TypeError('Only support 4D or 3D tensor')
    return img_gt, img_lq


class DenoiseMSSIDDUpiTrainValOfflineRAW2RAWDataset(Dataset):
    def __init__(self, opt):
        super(DenoiseMSSIDDUpiTrainValOfflineRAW2RAWDataset, self).__init__()
        self.opt = opt
        self.iso = self.opt['iso']
        self.sensor_list = self.opt['sensor_list']
        self.real_calib_sensor_num = len(self.sensor_list)
        self.img_size = self.opt['raw_img_size']
        self.crop_size = self.opt.get('crop_img_size', self.img_size)

        order_map = {'sensor_01': 0, 'sensor_02': 1, 'sensor_03': 2, 'sensor_04': 3, 'sensor_05': 4, 'sensor_06': 5}
        self.order_list = [order_map[s] for s in self.sensor_list]

        img_dir = self.opt['image_root']
        iso_tag = f'{self.iso[0]}_{self.iso[1]}'
        self.img_dir = os.path.join(img_dir, iso_tag)
        print(f'prepare dataset with iso {iso_tag} in {self.img_dir}.')
        if not os.path.exists(self.img_dir):
            raise NotImplementedError
    
        meta_data_path = os.path.join(self.img_dir, 'meta_data.pkl')
        with open(meta_data_path, 'rb') as f:
            self.meta_data = pickle.load(f)

        self.imnames = [i.split('.')[0] for i in os.listdir(os.path.join(self.img_dir, self.sensor_list[0], 'input_crops'))]
        self.imgs = [i+'.raw' for i in self.imnames]
        self.dataset_length = len(self.imgs)
        self.img_id_list = [i for i in range(len(self.imgs))] 
        print(f'Dataset patch number: {self.dataset_length}')

    def _read_raw(self, file_name):
        raw = np.fromfile(file_name, dtype=np.uint16)
        raw = raw.reshape(4, self.img_size, self.img_size)
        raw = raw / 4096.0
        return raw

    def __getitem__(self, index):
        noisy_images1, clean_images = [], []
        camera_ids = []
        awbs, cam2rgbs, rgb_gains = [], [], []
        cur_img_raw = self.imgs[index]
        cur_img_name = self.imnames[index]
        cur_meta_data = self.meta_data[cur_img_name]
        for i in range(self.real_calib_sensor_num):
            noisy_images1.append(torch.from_numpy(self._read_raw(os.path.join(self.img_dir, self.sensor_list[i], 'input_crops', cur_img_raw))).float())
            clean_images.append(torch.from_numpy(self._read_raw(os.path.join(self.img_dir, self.sensor_list[i], 'gt_crops', cur_img_raw))).float())
            camera_ids.append(i)
        for ord in self.order_list:
            awbs.append(torch.from_numpy(cur_meta_data['awb'][ord, :, :]).float())
            cam2rgbs.append(torch.from_numpy(cur_meta_data['ccm'][ord, :, :]).float())
            rgb_gains.append(torch.from_numpy(cur_meta_data['dgain'][ord, :]).float())
            
        noisy_images1 = torch.stack(noisy_images1, dim=0)   # sn,4,256,256
        clean_images = torch.stack(clean_images, dim=0)   # sn,4,256,256

        if self.img_size != self.crop_size:
            assert self.crop_size < self.img_size
            clean_images, noisy_images1 = paired_raw_random_crop(clean_images, noisy_images1, self.crop_size)

        cam2rgbs = torch.stack(cam2rgbs, dim=0)   # sn,3,3
        awbs = torch.stack(awbs, dim=0)    # sn,3,3
        rgb_gains = torch.stack(rgb_gains, dim=0)   # sn,1

        return {'lq': noisy_images1,  # 4,256,256  -> 6,4,256,256 -> 16,6,4,256,256 -> 96,4,256,256
            "gt": clean_images,
            "cid": torch.from_numpy(np.array(camera_ids)),
            "iid": torch.from_numpy(np.array(index)),
            "lq_path": cur_img_name,
            "awb": awbs,
            "cam2rgb": cam2rgbs,
            "rgb_gain": rgb_gains,
        }

    def __len__(self):
        return self.dataset_length


if __name__ == "__main__":
    opt = {
        'iso': [50, 3200],
        'sensor_list': ['sensor_01', 'sensor_02', 'sensor_03', 'sensor_04', 'sensor_05', 'sensor_06'],
        'raw_img_size': 256,
        'image_root': '/xxx/data/SIDD/MSSIDD_RAW2RAW/',
    }
    dataset = DenoiseMSSIDDUpiTrainValOfflineRAW2RAWDataset(opt=opt)
    item = dataset[0]