from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from trainers.BasicTrainer import BasicTrainer
import sys,shutil,os
import torchvision
import pytorch_iou
import pytorch_ssim
from SLBR.src.utils.osutils import mkdir_p, isfile, isdir, join
from tensorboardX import SummaryWriter
from math import log10
import skimage.io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import recall_score, accuracy_score
from progress.bar import Bar
import sys,time,os
sys.path.append('./SLBR')
sys.path.append('./MAT')
import torch.nn.functional as F
from evaluation import AverageMeter, compute_IoU, FScore, compute_RMSE
import SLBR.pytorch_ssim as pytorch_ssim
from SLBR.src.utils.osutils import mkdir_p, isfile, isdir, join
from SLBR.src.utils.parallel import DataParallelModel, DataParallelCriterion
from SLBR.src.utils.losses import l1_relative
import os
from torch import nn
import torch

from modules.TAWR import TAWR
from modules.Discriminator import Discriminator

class Losses(nn.Module):
    def __init__(self, argx):
        super(Losses, self).__init__()
        self.args = argx
        self.l1, self.masked_l1  = nn.L1Loss(), l1_relative
        self.wbce, self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.args.wm_bce_weight])), nn.BCEWithLogitsLoss() 
            
    def disc_loss(self, logits_real, logits_fake):        
        loss = (self.bce(logits_real, torch.ones_like(logits_real)) + 
                self.bce(logits_fake, torch.zeros_like(logits_fake))) / 2
        return loss
    
    def gan_loss(self, logits_fake):        
        return self.bce(logits_fake, torch.ones_like(logits_fake))   

    def forward(self, hardmask_pred, hardmask_gt, 
                      wm_pred, wm_gt, 
                      rec_out, disc_pred,
                      refine_out, original_gt, 
                      softmask_pred, softmask_gt, threshold=0.5):
        
        
        # reconstruction losses        
        rec_loss = self.args.lambda_l1_rec * self.masked_l1(rec_out, original_gt, hardmask_gt) + self.l1(rec_out, original_gt)
        rec_wm_loss = self.l1(wm_pred, wm_gt) # coarse stage        
        
        hardmask_pred_binary = (hardmask_pred.detach()>threshold).int()
        
        refine_loss = self.args.lambda_l1_rec * self.masked_l1(refine_out, original_gt, hardmask_pred_binary) + \
                      self.l1(refine_out, original_gt) # refinement stage

        softmask_loss = self.l1(softmask_pred, softmask_gt)
#         hardmask_loss = 25*self.l1(hardmask_pred, hardmask_gt)
        hardmask_loss = 10*self.wbce(hardmask_pred, hardmask_gt)
        
        gan_loss = self.gan_loss(disc_pred)

        return (rec_loss, rec_wm_loss, refine_loss, softmask_loss, hardmask_loss, gan_loss)



# based on SLBR codebase
class TAWRTrainer(BasicTrainer):
    def __init__(self, device=torch.device('cuda'), datasets =(None,None), models = None, args = None, **kwargs):        
        self.args = args        
        # create model
        print("==> creating model ")
        self.model = TAWR().to(device)
        self.disc = Discriminator().to(device)
        print("==> creating model [Finish]")
       
        self.train_loader, self.val_loader = datasets
        
        self.title = args.name
        self.args.checkpoint = os.path.join(args.checkpoint, self.title)
        self.device = device
        
         # create checkpoint dir
        if not isdir(self.args.checkpoint):
            mkdir_p(self.args.checkpoint)

        
#         if not self.args.evaluate:
        self.writer = SummaryWriter(self.args.checkpoint+'/'+'ckpt')
        
        self.best_acc = 0
        self.is_best = False
        self.current_epoch = 0
        self.metric = -100000
        self.loss = Losses(self.args)
        self.model.to(self.device)
        self.disc.to(self.device)
        self.loss.to(self.device)

        print('==> Total params TAWR: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        print('==> Total params Discriminator: %.2fM' % (sum(p.numel() for p in self.disc.parameters())/1000000.0))
        print('==> Total devices: %d' % (torch.cuda.device_count()))
        print('==> Current Checkpoint: %s' % (self.args.checkpoint))
        

        self.set_optimizers()
        if self.args.resume != '':
            self.resume(self.args.resume)
            
        
    def set_optimizers(self):
        self.optimizer_model = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                          lr=self.args.lr,
                                          betas=(self.args.beta1,self.args.beta2),
                                          weight_decay=self.args.weight_decay)  
        
        self.optimizer_disc = torch.optim.Adam(filter(lambda p: p.requires_grad, self.disc.parameters()), 
                                               lr=self.args.dlr,
                                               betas=(self.args.beta1,self.args.beta2),
                                               weight_decay=self.args.weight_decay) 

    def zero_grad_all(self):
        self.optimizer_model.zero_grad()
        self.optimizer_disc.zero_grad()
        
    def step_all(self):
        self.optimizer_model.step()
        self.optimizer_disc.step()
        
    def disc_grad(self):       
        
        self.model.eval()
        self.disc.train() 
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.disc.parameters():
            param.requires_grad = True
        
    def model_grad(self):  
        
        self.model.train()
        self.disc.eval()   
        
        for param in self.model.parameters():
            param.requires_grad = True               
            
        for param in self.disc.parameters():
            param.requires_grad = False

            
    def train(self,epoch):

        self.current_epoch = epoch
        
        #rec_loss, refine_loss, softmask_loss, hardmask_loss, gan_loss, disc_loss
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_rec_meter = AverageMeter()
        loss_refine_meter = AverageMeter()
        loss_softmask_meter = AverageMeter()
        loss_hardmask_meter = AverageMeter()
        loss_gan_meter = AverageMeter()
        loss_disc_meter = AverageMeter()
        loss_total_meter = AverageMeter()
        f1_meter = AverageMeter()
        # switch to train mode
        self.model.train()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.downsample = nn.Upsample(scale_factor=.25, mode='bilinear')
        end = time.time()
        bar = Bar('Processing TAWR', max=len(self.train_loader))
        for i, batches in enumerate(self.train_loader):
            current_index = len(self.train_loader) * epoch + i

            inputs = batches['image'].float().to(self.device)
            original_gt = batches['target'].float().to(self.device)
            hardmask_gt = batches['mask'].float().to(self.device)
            wm_gt =  batches['wm'].float().to(self.device)
            softmask_gt = batches['alpha'].float().to(self.device)
            img_path = batches['img_path']
            
            # first eval model
            # set model training mode + gradient calc
            self.zero_grad_all()  
            self.model_grad()
            outputs = self.model(inputs)
            original_pred, rec_out, wm_pred, hardmask_pred, softmask_pred = outputs
            
            disc_pred = self.disc(torch.cat([original_pred, hardmask_gt],dim=1))
            
            tawr_loss = self.loss(hardmask_pred, hardmask_gt, 
                                             wm_pred, wm_gt, 
                                             rec_out, disc_pred, 
                                             original_pred, original_gt, 
                                             softmask_pred, softmask_gt)
            
            (rec_loss, rec_wm_loss, refine_loss, softmask_loss, hardmask_loss, gan_loss) = tawr_loss
            
            model_loss = self.args.lambda_l1*(rec_loss + refine_loss + rec_wm_loss + softmask_loss) \
                        + self.args.lambda_mask*(hardmask_loss) + self.args.lambda_gan*(gan_loss)
            
            model_loss.backward()
            self.optimizer_model.step()
            
            # now update the discriminator
            self.disc_grad()
            self.zero_grad_all() 
            disc_real = self.disc(torch.cat([inputs,hardmask_gt],dim=1))
            disc_pred = self.disc(torch.cat([original_pred.detach(), hardmask_gt],dim=1))            
            disc_loss = self.loss.disc_loss(disc_real, disc_pred)            
            disc_loss.backward()  
            self.optimizer_disc.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            loss_rec_meter.update(rec_loss.item())
            loss_refine_meter.update(refine_loss.item())
            loss_softmask_meter.update(softmask_loss.item())
            loss_hardmask_meter.update(hardmask_loss.item())
            loss_gan_meter.update(gan_loss.item())
            loss_disc_meter.update(disc_loss.item())
            loss_total_meter.update(model_loss.item())
            #f1_meter.update(FScore(hardmask_pred, self.downsample(hardmask_gt)).item())
            f1_meter.update(FScore(hardmask_pred, hardmask_gt).item())
            # measure accuracy and record loss
#             losses_meter.update(coarse_loss.item(), inputs.size(0))


            # plot progress
            suffix  = "({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | loss L1 rec: {loss_rec:.4f} | loss L1 ref: {loss_ref:.4f} | loss L1 softmask: {loss_softmask:.4f} | loss hardmask: {loss_hardmask:.4f} | loss gan: {loss_gan:.4f} | loss disc: {loss_disc:.4f} | loss total: {loss_total:.4f} | mask F1: {mask_f1:.4f}".format(
                        batch=i + 1,
                        size=len(self.train_loader),
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_rec=loss_rec_meter.avg,
                        loss_ref=loss_refine_meter.avg,
                        loss_softmask=loss_softmask_meter.avg,
                        loss_hardmask=loss_hardmask_meter.avg,
                        loss_gan=loss_gan_meter.avg,
                        loss_disc=loss_disc_meter.avg,
                        loss_total=loss_total_meter.avg,
                        mask_f1=f1_meter.avg)
        
            if current_index % 100 == 0:
                print(suffix)
                self.record('train/loss_L1_rec', loss_rec_meter.avg, current_index)
                self.record('train/loss_L1_ref', loss_refine_meter.avg, current_index)
                self.record('train/loss_softmask', loss_softmask_meter.avg, current_index)
                self.record('train/loss_hardmask', loss_hardmask_meter.avg, current_index)
                self.record('train/loss_gan', loss_gan_meter.avg, current_index)
                self.record('train/loss_disc', loss_disc_meter.avg, current_index)
                self.record('train/loss_model', loss_total_meter.avg, current_index)
                self.record('train/mask_F1', f1_meter.avg, current_index)

                #mask_pred = (torch.sigmoid(hardmask_pred) > 0.5).int()
                mask_pred = (hardmask_pred > 0.5).int()
                
                bg_pred = original_pred*mask_pred + (1-mask_pred)*inputs
                show_size = 5 if inputs.shape[0] > 5 else inputs.shape[0]
                self.image_display = torch.cat([
                    inputs[0:show_size].detach().cpu(),             # input image
                    original_gt[0:show_size].detach().cpu(),                        # ground truth
                    bg_pred.clamp(0,1)[0:show_size].detach().cpu(),
                    mask_pred[0:show_size].detach().cpu().repeat(1,3,1,1),       # refine out
                    softmask_pred[0:show_size].detach().cpu().repeat(1,3,1,1),
                    hardmask_gt[0:show_size].detach().cpu().repeat(1,3,1,1),
                    original_pred.clamp(0,1)[0:show_size].detach().cpu(),
                    wm_pred.clamp(0,1)[0:show_size].detach().cpu(),
                    rec_out.clamp(0,1)[0:show_size].detach().cpu()
                ],dim=0)
                image_dis = torchvision.utils.make_grid(self.image_display, nrow=show_size)
                self.writer.add_image('Image', image_dis, current_index)
            del outputs
            
    def validate(self, epoch):

        self.current_epoch = epoch
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        loss_mask_meter = AverageMeter()
        psnr_meter = AverageMeter()
        fpsnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        rmse_meter = AverageMeter()
        rmsew_meter = AverageMeter()        
        recall_mask_meter = AverageMeter()        
        accuracy_mask_meter = AverageMeter()        
        l2_softmask_meter = AverageMeter()

        iou_meter = AverageMeter()
        f1_meter = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        start = time.time()
        b_start = time.time()
        bar = Bar('Processing TAWR', max=len(self.val_loader))
        with torch.no_grad():
            for i, batches in enumerate(self.val_loader):
                
                current_index = len(self.val_loader) * epoch + i

                inputs = batches['image'].float().to(self.device)
                original_gt = batches['target'].float().to(self.device)
                hardmask_gt = batches['mask'].float().to(self.device)
#                 wm_gt =  batches['wm'].float().to(self.device)
#                 softmask_gt = batches['alpha'].float().to(self.device)
#                 img_path = batches['img_path']
                # alpha_gt = batches['alpha'].float().to(self.device)

                outputs = self.model(inputs)
                original_pred, rec_out, wm_pred, hardmask_pred, softmask_pred = outputs
                mask_pred = (hardmask_pred > 0.5).int()
                bg_pred = original_pred*mask_pred + (1-mask_pred)*inputs
                
                accuracy = accuracy_score(hardmask_gt.int().cpu().flatten(), mask_pred.cpu().flatten())
                recall = recall_score(hardmask_gt.int().cpu().flatten(), mask_pred.cpu().flatten(), zero_division=1)
                eps = 1e-6
                psnr = 10 * log10(1 / F.mse_loss(bg_pred,original_gt).item()) 
                fmse = F.mse_loss(bg_pred*hardmask_gt, original_gt*hardmask_gt, reduction='none').sum(dim=[1,2,3]) / (hardmask_gt.sum(dim=[1,2,3])*3+eps)
                fpsnr = 10 * torch.log10(1 / fmse).mean().item()
                ssim = pytorch_ssim.ssim(bg_pred,original_gt)

                psnr_meter.update(psnr, inputs.size(0))
                fpsnr_meter.update(fpsnr, inputs.size(0))
                ssim_meter.update(ssim, inputs.size(0))
                rmse_meter.update(compute_RMSE(bg_pred,original_gt,hardmask_gt),inputs.size(0))
                rmsew_meter.update(compute_RMSE(bg_pred,original_gt,hardmask_gt,is_w=True), inputs.size(0))
#                 l2_softmask_meter.update(F.mse_loss(softmask_pred, softmask_gt, reduction='mean'))
                iou = compute_IoU(hardmask_gt, mask_pred)
                iou_meter.update(iou, inputs.size(0))
                recall_mask_meter.update(recall)
                accuracy_mask_meter.update(accuracy)
                f1 = FScore(hardmask_gt, hardmask_pred).item()
                f1_meter.update(f1, inputs.size(0))
                # measure elapsed time
                batch_time.update(time.time() - b_start)


                suffix  = '({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | PSNR: {psnr:.4f} | fPSNR: {fpsnr:.4f} | SSIM: {ssim:.4f} | Softmask L2: {soft:.4f} | RMSE: {rmse:.4f} | RMSEw: {rmsew:.4f} | IoU: {iou:.4f} | F1: {f1:.4f}'.format(
                                batch=i + 1,
                                size=len(self.val_loader),
                                data=data_time.val,
                                bt=batch_time.val,
                                total=bar.elapsed_td,
                                eta=bar.eta_td,
                                psnr=psnr_meter.avg,
                                fpsnr=fpsnr_meter.avg,
                                ssim=ssim_meter.avg,
                                soft=l2_softmask_meter.avg,
                                rmse=rmse_meter.avg,
                                rmsew=rmsew_meter.avg,
                                iou=iou_meter.avg,
                                f1=f1_meter.avg
                                )

                if i%100 == 0:
                    
                    show_size = 5 if inputs.shape[0] > 5 else inputs.shape[0]
                    self.image_display = torch.cat([
                        inputs[0:show_size].detach().cpu(),             # input image
                        original_gt[0:show_size].detach().cpu(),        # ground truth
                        bg_pred.clamp(0,1)[0:show_size].detach().cpu(),
                        hardmask_gt[0:show_size].detach().cpu().repeat(1,3,1,1),
                        mask_pred[0:show_size].detach().cpu().repeat(1,3,1,1),       # refine out
                        wm_pred.clamp(0,1)[0:show_size].detach().cpu()
                    ],dim=0)
                    image_dis = torchvision.utils.make_grid(self.image_display, nrow=show_size)
                    self.writer.add_image('Image', image_dis, current_index)
                    print(suffix)
                # bar.next()
                
                b_start = time.time()
        print("Total:")
        print(suffix)
        bar.finish()
        
        print("Iter:%s,losses:%s,PSNR:%.4f,SSIM:%.4f"%(epoch, losses_meter.avg,psnr_meter.avg,ssim_meter.avg))
        self.record('val/rmse', rmse_meter.avg, epoch)
#         self.record('val/rmsew', rmsew_meter.avg, epoch)
        self.record('val/accuracy_mask', accuracy_mask_meter.avg, epoch)
        self.record('val/recall_mask', recall_mask_meter.avg, epoch)
        self.record('val/f1_mask', f1_meter.avg, epoch)
        self.record('val/iou_meter', iou_meter.avg, epoch)
#         self.record('val/l2_softmask_meter', l2_softmask_meter.avg, epoch)
        self.record('val/PSNR', psnr_meter.avg, epoch)
        self.record('val/SSIM', ssim_meter.avg, epoch)
        self.record('val/time', time.time()-start, epoch)
        # batch size not accounted for as it cancels on calculating per-sample time
        self.record('val/time_per_sample', (time.time()-start)/(len(self.val_loader)), epoch)
        self.metric = psnr_meter.avg

        self.model.train()
    
            
    def resume(self, resume_path):
        # if isfile(resume_path):
        if not os.path.exists(resume_path):
            resume_path = os.path.join(self.args.checkpoint, 'checkpoint.pth.tar')
        if not os.path.exists(resume_path):
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))

        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)

        if self.args.start_epoch == 0:
            self.args.start_epoch = current_checkpoint['epoch']
        self.metric = current_checkpoint['best_acc']
        # items = list(current_checkpoint['state_dict'].keys())

        ## restore the learning rate
        lr = self.args.lr
        for epoch in self.args.schedule:
            if epoch <= self.args.start_epoch:
                lr *= self.args.gamma
                
                
        if current_checkpoint['optimizer_model'] is not None:
            self.optimizer_model.load_state_dict(current_checkpoint['optimizer_model'])
            print("=> restored model optimizer from checkpoint")
        if current_checkpoint['optimizer_disc'] is not None:
            self.optimizer_disc.load_state_dict(current_checkpoint['optimizer_disc'])
            print("=> restored discriminator optimizer from checkpoint")
        
#         optimizers = [getattr(self.model, attr) for attr in dir(self.model) if  attr.startswith("optimizer") and getattr(self.model, attr) is not None]
#         for optimizer in optimizers:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
        
        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint['state_dict_model'], strict=True)
        self.disc.load_state_dict(current_checkpoint['state_dict_disc'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, current_checkpoint['epoch']))    
            
    def save_checkpoint(self,filename='checkpoint.pth.tar', snapshot=None):
        is_best = True if self.best_acc < self.metric else False

        if is_best:
            self.best_acc = self.metric

        state = {
                    'epoch': self.current_epoch + 1,
                    'state_dict_model': self.model.state_dict(),
                    'state_dict_disc': self.disc.state_dict(),
                    'best_acc': self.best_acc,
                    'optimizer_model' : self.optimizer_model.state_dict() if self.optimizer_model else None,
                    'optimizer_disc' : self.optimizer_disc.state_dict() if self.optimizer_disc else None,
                }

        filepath = os.path.join(self.args.checkpoint, filename)
        torch.save(state, filepath)

        if snapshot and state['epoch'] % snapshot == 0:
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'checkpoint_{}.pth.tar'.format(state.epoch)))
        
        if is_best:
            self.best_acc = self.metric
            print('Saving Best Metric with PSNR:%s'%self.best_acc)
            if not os.path.exists(self.args.checkpoint): os.makedirs(self.args.checkpoint)
            shutil.copyfile(filepath, os.path.join(self.args.checkpoint, 'model_best.pth.tar'))
            
    def resume(self, resume_path):
        # if isfile(resume_path):
        if not os.path.exists(resume_path):
            resume_path = os.path.join(self.args.checkpoint, 'checkpoint.pth.tar')
        if not os.path.exists(resume_path):
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))

        print("=> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)

        if self.args.start_epoch == 0:
            self.args.start_epoch = current_checkpoint['epoch']
        self.metric = current_checkpoint['best_acc']
#         items = list(current_checkpoint['state_dict'].keys())

        ## restore the learning rate
        lr = self.args.lr
        for epoch in self.args.schedule:
            if epoch <= self.args.start_epoch:
                lr *= self.args.gamma
        if current_checkpoint['optimizer_model'] is not None:
            self.optimizer_model.load_state_dict(current_checkpoint['optimizer_model'])
            print("=> restored model optimizer from checkpoint")
        if current_checkpoint['optimizer_disc'] is not None:
            self.optimizer_disc.load_state_dict(current_checkpoint['optimizer_disc'])
            print("=> restored discriminator optimizer from checkpoint")
        
#         optimizers = [getattr(self.model, attr) for attr in dir(self.model) if  attr.startswith("optimizer") and getattr(self.model, attr) is not None]
#         for optimizer in optimizers:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
        
        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint['state_dict_model'], strict=True)
        self.disc.load_state_dict(current_checkpoint['state_dict_disc'], strict=True)
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_path, current_checkpoint['epoch']))