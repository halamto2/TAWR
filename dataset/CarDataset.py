from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
import random

class CarDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, args):
        
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

        self.args = args
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

        self.imageJ_path=osp.join(self.root,'watermarked','%s.jpg')
        self.imageI_path=osp.join(self.root,'original','%s.jpg')
        self.mask_path=osp.join(self.root,'hard_mask','%s.png')
        self.alpha_path=osp.join(self.root,'soft_mask','%s.png')
        self.W_path=osp.join(self.root,'watermark','%s.png')
        
        self.ids = list()
        for file in os.listdir(self.root+'/original'):
            self.ids.append(file.strip('.jpg'))
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        
        
       
    def __len__(self):
        return len(self.ids)
    
    def get_sample(self, index):
#         print("getting sample", index)
        img_id = self.ids[index]
#         print("loaded filepath", index)
        # img_id = self.corrupt_list[index % len(self.corrupt_list)].split('.')[0]
        img_J = Image.open(self.imageJ_path%img_id).convert('RGB').resize((256,256))
#         print("loaded J", index)
        img_I = Image.open(self.imageI_path%img_id).convert('RGB').resize((256,256))
#         print("loaded I", index)
        w = Image.open(self.W_path%img_id).resize((256,256))
#         print("loaded W", index)
        w = np.array(w)
#         print("w in array", index)
        alpha = w[:,:,3]
#         print("alpha from w", index)
        valid = alpha>0
#         print("hardmask done", index)
        rgb = w[:,:,:3]
#         print("rgb wm done", index)
        w = np.zeros_like(rgb)+rgb[:,:,:3]*np.stack((valid,)*3,axis=2)
#         print("processed imgs, loading mask", index) 
        mask = 255-np.array(Image.open(self.mask_path%img_id).resize((256,256)))
#         print("processed imgs, loading alpha mask", index) 
        alpha = 255-np.array(Image.open(self.alpha_path%img_id).resize((256,256)))
        
        mask = mask.astype(np.float32) / 255.
        alpha = alpha.astype(np.float32) / 255.
        
#         print("processed, returning", index)
        
        return {'J': np.array(img_J), 
                'I': np.array(img_I), 
                'watermark': w, 
                'mask':mask, 
                'alpha':alpha, 
                'img_path':self.imageJ_path%img_id }


    def __getitem__(self, index):
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

    def check_sample_types(self, sample):
        assert sample['J'].dtype == 'uint8'
        assert sample['I'].dtype == 'uint8'
        assert sample['watermark'].dtype == 'uint8'

    def augment_sample(self, sample):
        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True
        return aug_output['mask'].sum() > 100



