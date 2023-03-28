from __future__ import print_function, absolute_import

import argparse
import torch,time,os

torch.backends.cudnn.benchmark = True
import sys
from options import Options

from SLBR.src.utils.misc import save_checkpoint, adjust_learning_rate

import datasets as datasets
import numpy as np

from TAWRTrainer import TAWRTrainer

def main(args):
    args.seed = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset == 'clwd':
        dataset_func = datasets.CLWDDataset
    elif args.dataset == 'cars':
        dataset_func = datasets.CarDataset
    elif args.dataset == 'lvw':
        dataset_func = datasets.LVWDataset
    else:
        raise ValueError("Not known dataset:\t{}".format(args.dataset))

    train_loader = torch.utils.data.DataLoader(dataset_func('train',args),batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    lr = args.lr
    data_loaders = (train_loader,val_loader)

    model = TAWRTrainer(datasets=data_loaders, args=args)
    print('============================ Initization Finish && Training Start =============================================')

    print("record")
    print("eval")
    model.validate(0)
    model.flush()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('==================================== WaterMark Removal =============================================')
    print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
    print('==> {:50}: {:<}'.format("USE GPU",os.environ['CUDA_VISIBLE_DEVICES']))
    print('==================================== Stable Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) == ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) == parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Changed Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) != ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) != parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Start Init Model  ===============================================')
    main(args)
    print('==================================== FINISH WITHOUT ERROR =============================================')
