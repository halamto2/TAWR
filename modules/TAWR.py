import torch.nn as nn
from modules.WatermarkRemover import WatermarkRemover
from modules.WatermarkRefiner import WatermarkRefiner


class TAWR(nn.Module):
    def __init__(self):
        super().__init__()
        self.wm_rem = WatermarkRemover()
        self.wm_ref = WatermarkRefiner()
    
    def forward(self, img):
        rec, wm, hardmask, softmask = self.wm_rem(img)
        mask = (hardmask>0.5).int()
        out = self.wm_ref(img*(1-mask) + rec*mask)
        return out, rec, wm, hardmask, softmask