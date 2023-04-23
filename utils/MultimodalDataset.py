import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image



class MMDataset(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir_2d, data_dir_pc, datainfo_path, transform, crop_size = 224, img_length_read = 4, patch_length_read = 6, npoint = 2048, is_train = True):
        super(MMDataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        self.ply_name = dataInfo[['name']]
        self.ply_mos = dataInfo['mos']
        self.crop_size = crop_size
        self.data_dir_2d = data_dir_2d
        self.transform = transform
        self.img_length_read = img_length_read
        self.patch_length_read = patch_length_read
        self.npoint = npoint
        self.data_dir_pc = data_dir_pc
        self.length = len(self.ply_name)
        self.is_train = is_train

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        img_name = self.ply_name.iloc[idx,0] 
        frames_dir = os.path.join(self.data_dir_2d, img_name)

        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size
       
        img_length_read = self.img_length_read       
        transformed_img = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        # read images
        img_read_index = 0
        for i in range(img_length_read):
            # load images
            imge_name = os.path.join(frames_dir, str(i) + '.png')
            if os.path.exists(imge_name):
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                transformed_img[i] = read_frame

                img_read_index += 1
            else:
                print(imge_name)
                print('Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                transformed_img[j] = transformed_img[img_read_index-1]

        # read pc
        patch_length_read = self.patch_length_read
        npoint = self.npoint
        selected_patches = torch.zeros([patch_length_read, 3, npoint])
        path = os.path.join(self.data_dir_pc,self.ply_name.iloc[idx,0].split('.')[0]+'.npy')
        points = list(np.load(path))
        # randomly select patches during the training stage
        if self.is_train:
            random_patches = random.sample(points, patch_length_read)
        else:
            random_patches = points
        for i in range(patch_length_read):
            selected_patches[i] = torch.from_numpy(random_patches[i]).transpose(0,1)

        y_mos = self.ply_mos.iloc[idx] 
        y_label = torch.FloatTensor(np.array(y_mos))

       
        return transformed_img, selected_patches, y_label