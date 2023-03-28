class Options():

    def __init__(self):
        pass

    def init(self, parser):                                   
        parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=30, type=int, metavar='N',
                            help='number of total epochs to run')
        
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        
        parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                            help='train batchsize')
        parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                            help='test batchsize')
        
        parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                            metavar='W', help='weight decay (default: 0)')
        
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,metavar='LR', help='initial learning rate')
        parser.add_argument('--dlr', '--dlearning-rate', default=1e-3, type=float, help='initial learning rate')
        
        parser.add_argument('--beta1', default=0.9, type=float, help='initial learning rate')
        parser.add_argument('--beta2', default=0.999, type=float, help='initial learning rate')
        parser.add_argument('--momentum', default=0, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR is multiplied by gamma on schedule.')
        # Data processing
        parser.add_argument('--lambda_l1', type=float, default=4, help='the weight of L1.')

        parser.add_argument('--lambda_mask', default=1, type=float,help='mask loss')
        
        # Miscs
        parser.add_argument('--dataset_dir', default='/PATH_TO_DATA_FOLDER/', type=str, metavar='PATH')
        
        parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        
        parser.add_argument('--input-size', default=256, type=int, metavar='N',
                            help='train batchsize')
       
        parser.add_argument('--gpu_id',default='0',type=str)

        parser.add_argument('--dataset', default='clwd',type=str, help='train batchsize')
        parser.add_argument('--name', default='v2',type=str, help='train batchsize')
                
        parser.add_argument('--lambda_gan', default=0, type=float,
                            help='gan loss')
        
        parser.add_argument('--lambda_l1_rec', default=10, type=float,
                            help='rec l1 loss lambda')
        
        parser.add_argument('--wm_bce_weight', default=10, type=float,
                            help='bcewithlogitsloss pos_weight for wm detection')

        return parser