import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import sys

from PVT.classification.pvt_v2 import pvt_v2_b0, PyramidVisionTransformerV2, partial, Block, OverlapPatchEmbed
from MAT.networks.basic_module import Conv2dLayer
from mmseg.models.decode_heads import SegformerHead
# from MySegformerHead import MySegformerHead

class MultifeaturePyramidVisionTransformerV2(PyramidVisionTransformerV2):
    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

class PVTDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params=params
        self.embeds = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.patch_size = 3
        
        for i in range(3,0, -1):
            embed = OverlapPatchEmbed(img_size=params['img_size'] // (2 ** (i + 2)),
                                      patch_size=3,
                                      stride = 2,
                                      in_chans=params['embed_dims'][i],
                                      embed_dim=params['embed_dims'][i-1])
            
            embed.proj = Conv2dLayer(in_channels=params['embed_dims'][i], 
                                        out_channels=params['embed_dims'][i-1], 
                                        kernel_size=self.patch_size, 
                                        up=2)
            self.embeds.append(embed)

            block = nn.ModuleList([Block(dim=params['embed_dims'][i-1], 
                                         num_heads=params['num_heads'][i-1], 
                                         mlp_ratio=params['mlp_ratios'][i-1], 
                                         qkv_bias=params['qkv_bias'], 
                                         sr_ratio=params['sr_ratios'][i-1]) for j in range(params['depths'][i-1])])
            self.decoder_blocks.append(block)

            self.norms.append(nn.LayerNorm(params['embed_dims'][i-1]))
            
    def forward(self, encoder_out):
        x = encoder_out[-1]
        B = x.shape[0]
        decoder_out = [x]
        for i in range(len(self.decoder_blocks)):
            x, H, W = self.embeds[i](x)
            for blk in self.decoder_blocks[i]:
                x = blk(x,H,W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x += encoder_out[len(self.decoder_blocks) - i - 1]
            decoder_out.append(x)
        return decoder_out
    
class HardMaskDecoder(nn.Module):
    def __init__(self, segformer_params):
        super().__init__()
        self.seghead = SegformerHead(**segformer_params)

    def forward(self, x):
        return self.seghead(x)

    
class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            Conv2dLayer(in_channels=64, out_channels=64, activation = 'lrelu',  kernel_size=3),
            nn.BatchNorm2d(64),
            Conv2dLayer(in_channels=64, out_channels=32, activation = 'lrelu',  kernel_size=3, up=2),
            nn.BatchNorm2d(32),
            Conv2dLayer(in_channels=32, out_channels=32, activation = 'lrelu',  kernel_size=3),
            nn.BatchNorm2d(32),
            Conv2dLayer(in_channels=32, out_channels=16, activation = 'lrelu',  kernel_size=3, up=2),
            nn.BatchNorm2d(16),
            Conv2dLayer(in_channels=16, out_channels=16, activation = 'lrelu', kernel_size=3),
            nn.BatchNorm2d(16),
            Conv2dLayer(in_channels=16, out_channels=3, activation = 'relu', kernel_size=3)
        )

    def forward(self, x):
        return self.decoder(x)
    

class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = nn.Sequential(
            Conv2dLayer(in_channels=64, out_channels=64, activation = 'lrelu',  kernel_size=3),
            nn.BatchNorm2d(64),
            Conv2dLayer(in_channels=64, out_channels=32, activation = 'lrelu',  kernel_size=3, up=2),
            nn.BatchNorm2d(32),
            Conv2dLayer(in_channels=32, out_channels=32, activation = 'lrelu',  kernel_size=3),
            nn.BatchNorm2d(32),
            Conv2dLayer(in_channels=32, out_channels=16, activation = 'lrelu',  kernel_size=3, up=2),
            nn.BatchNorm2d(16)
        )
        
        self.out_hard = nn.Sequential(
            Conv2dLayer(in_channels=16, out_channels=16, activation = 'lrelu', kernel_size=3),
            nn.BatchNorm2d(16),
            Conv2dLayer(in_channels=16, out_channels=1, activation = 'relu', kernel_size=3)
        )
        
        self.out_soft = nn.Sequential(
            Conv2dLayer(in_channels=16, out_channels=16, activation = 'lrelu', kernel_size=3),
            nn.BatchNorm2d(16),
            Conv2dLayer(in_channels=16, out_channels=1, activation = 'relu', kernel_size=3)
        )

    def forward(self, x):
        x = self.decoder(x)
        return self.out_hard(x), self.out_soft(x)

    

class SoftMaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder = nn.Sequential(
            Conv2dLayer(in_channels=64, out_channels=64, activation = 'lrelu',  kernel_size=3),
            nn.BatchNorm2d(64),
            Conv2dLayer(in_channels=64, out_channels=32, activation = 'lrelu',  kernel_size=3, up=2),
            nn.BatchNorm2d(32),
            Conv2dLayer(in_channels=32, out_channels=32, activation = 'lrelu',  kernel_size=3),
            nn.BatchNorm2d(32),
            Conv2dLayer(in_channels=32, out_channels=16, activation = 'lrelu',  kernel_size=3, up=2),
            nn.BatchNorm2d(16),
            Conv2dLayer(in_channels=16, out_channels=16, activation = 'lrelu', kernel_size=3),
            nn.BatchNorm2d(16),
            Conv2dLayer(in_channels=16, out_channels=1, activation = 'relu', kernel_size=3)
        )
        
    def forward(self, x):
        return self.decoder(x)
    
class WatermarkRemover(nn.Module):
    
    DEFAULT_PVT = {
            'patch_size':4, 
            'embed_dims':[32, 64, 160, 256], 
            'num_heads':[1, 2, 5, 8], 
            'mlp_ratios':[8, 8, 4, 4], 
            'qkv_bias':True,
            'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
            'depths':[2, 2, 2, 2], 
            'sr_ratios':[8, 4, 2, 1],
            'img_size': 256,
            'encoder_weights': './pretrained_weights/pvt_v2_b0.pth'
        }
    
    DEFAULT_PVT_B1 = dict(patch_size=4, 
                          embed_dims=[64, 128, 320, 512], 
                          num_heads=[1, 2, 5, 8], 
                          mlp_ratios=[8, 8, 4, 4], 
                          qkv_bias=True,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                          depths=[2, 2, 2, 2], 
                          sr_ratios=[8, 4, 2, 1],
                          img_size=256,
                          encoder_weights='./pretrained_weights/pvt_v2_b1.pth')
    
    DEFAULT_SEGFORMER = dict(in_channels=[64, 128, 320, 512],
                             in_index=[0, 1, 2, 3][::-1],
                             channels=32,
                             dropout_ratio=0.1,
                             num_classes=2,
                             out_channels=1, # should i use this?
                             threshold=0.5,
                             align_corners=False)
    
    
    def __init__(self, pvt_params=DEFAULT_PVT_B1, segformer_params=DEFAULT_SEGFORMER):
        super().__init__()
        self.pvt_params = pvt_params.copy()
        encoder_weights_file = self.pvt_params.pop('encoder_weights')        
        self.pvt_encoder = MultifeaturePyramidVisionTransformerV2(**self.pvt_params)        
        print(self.pvt_encoder.load_state_dict(torch.load(encoder_weights_file)))
#         self.pvt_encoder.forward = self.pvt_encoder.forward_features
        self.pvt_decoder = PVTDecoder(self.pvt_params)
#         self.mask_decoder = MaskDecoder()
        self.softmask_decoder = SoftMaskDecoder()
        self.hardmask_decoder = SegformerHead(**segformer_params)
        self.wm_decoder = ImageDecoder()
        self.rec_decoder = ImageDecoder()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
    
    def forward(self, img):
        encoder_out = self.pvt_encoder(img)
        decoder_out = self.pvt_decoder(encoder_out)
        wm_out = self.wm_decoder(decoder_out[-1])
        rec_out = self.rec_decoder(decoder_out[-1])
#         hardmask_out, softmask_out = self.mask_decoder(decoder_out)
        softmask_out, hardmask_out = self.softmask_decoder(decoder_out[-1]), self.hardmask_decoder(decoder_out)
        return rec_out, wm_out, self.upsample(hardmask_out), softmask_out
   