from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
import random

class CLWDDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, args):
        self.is_train = is_train == 'train'
        args.is_train = is_train == 'train'
        if args.is_train == True:
            self.root = args.dataset_dir + '/train/'
            # self.keep_background_prob = 0.01
            self.keep_background_prob = -1
        elif args.is_train == False:
            self.root = args.dataset_dir + '/test/' #'/test/'
            self.keep_background_prob = -1
            args.preprocess = 'resize'
            args.no_flip = True
            
        # Augmentataion?
        self.transform_norm=transforms.Compose([
            transforms.ToTensor()])
            # transforms.Normalize(
            #         # (0.485, 0.456, 0.406),
            #         # (0.229, 0.224, 0.225)
            #     (0.5,0.5,0.5),
            #     (0.5,0.5,0.5)
            # )])
       
        self.transform_tensor = transforms.ToTensor()

        self.imageJ_path=osp.join(self.root,'Watermarked_image','%s.jpg')
        self.imageI_path=osp.join(self.root,'Watermark_free_image','%s.jpg')
        self.mask_path=osp.join(self.root,'Mask','%s.png')
        self.alpha_path=osp.join(self.root,'Alpha','%s.png')
        self.W_path=osp.join(self.root,'Watermark','%s.png')

        self.ids = list()
        for file in os.listdir(self.root+'/Watermarked_image'):
            self.ids.append(file.strip('.jpg'))
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        
        
       
    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, index):
        img_id = self.ids[index]
        # img_id = self.corrupt_list[index % len(self.corrupt_list)].split('.')[0]
        img_J = cv2.imread(self.imageJ_path%img_id)
        img_J = cv2.cvtColor(img_J, cv2.COLOR_BGR2RGB)

        img_I = cv2.imread(self.imageI_path%img_id)
        img_I = cv2.cvtColor(img_I, cv2.COLOR_BGR2RGB)
        
        
        mask = cv2.imread(self.mask_path%img_id)        
        mask = mask[:, :, 0].astype(np.float32) / 255.
        
        if self.is_train:            
            alpha = cv2.imread(self.alpha_path%img_id)
            alpha = alpha[:, :, 0].astype(np.float32) / 255.
            w = cv2.imread(self.W_path%img_id)
            if w is None: print(self.W_path%img_id)
            w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
            return {'J': img_J, 'I': img_I, 'watermark': w, 'mask':mask, 'alpha':alpha, 'img_path':self.imageJ_path%img_id}
        else:
            return {'J': img_J, 'I': img_I, 'mask':mask, 'img_path':self.imageJ_path%img_id}


    def __getitem__(self, index):
        if self.is_train:
            sample = self.get_sample(index)
            self.check_sample_types(sample)
            sample = self.augment_sample(sample)

            J = self.transform_norm(sample['J'])
            I = self.transform_norm(sample['I'])
            w = self.transform_norm(sample['watermark'])

            mask = sample['mask'][np.newaxis, ...].astype(np.float32)
            mask = np.where(mask > 0.1, 1, 0).astype(np.uint8)
            alpha = sample['alpha'][np.newaxis, ...].astype(np.float32)

            data = {
                'image': J,
                'target': I,
                'wm': w,
                'mask': mask,
                'alpha':alpha,
                'img_path':sample['img_path']
            }
            return data
        else:
            sample = self.get_sample(index)
            self.check_sample_types(sample)
            sample = self.augment_sample(sample)

            J = self.transform_norm(sample['J'])
            I = self.transform_norm(sample['I'])

            mask = sample['mask'][np.newaxis, ...].astype(np.float32)
            mask = np.where(mask > 0.1, 1, 0).astype(np.uint8)

            data = {
                'image': J,
                'target': I,
                'mask': mask,
                'img_path':sample['img_path']
            }
            return data

    def check_sample_types(self, sample):
        assert sample['J'].dtype == 'uint8'
        assert sample['I'].dtype == 'uint8'
        if 'watermark' in sample:
            assert sample['watermark'].dtype == 'uint8'

    def augment_sample(self, sample):
        return sample


    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        return aug_output['mask'].sum() > 100



