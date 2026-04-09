import time
import os
import shutil
import random
import torch.autograd
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.metric_tool import get_mIoU
import numpy as np
import argparse

working_path = os.path.abspath('.')
from collections import OrderedDict
from utils.loss import LatentSimilarity
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter
random_number = random.randrange(1, 10000)
print(random_number)

# save_visuals It can be used to switch on and off and save the visualization

###################### Data and Model ########################
from models.SAM_Fusion4 import SAM_CD as Net
NET_NAME = ''
from datasets import data as RS
DATA_NAME = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    
    # training parameters
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--no-gpu', dest='gpu', action='store_false')
    parser.add_argument('--dev_id', type=int, default=0)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--predict_step', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--load_premodel', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=2137)
    
    # path parameters
    parser.add_argument('--working_path', type=str, default='.')
    parser.add_argument('--DATA_NAME', type=str, default='data')
    parser.add_argument('--NET_NAME', type=str, default='net')
    parser.add_argument('--chkpt_path', type=str, default='')
    
    args = parser.parse_args()
    
    args.pred_dir = os.path.join(args.working_path, 'results', args.DATA_NAME)
    args.chkpt_dir = os.path.join(args.working_path, 'checkpoints', args.NET_NAME, args.DATA_NAME)
    args.log_dir = os.path.join(args.working_path, 'logs', args.NET_NAME, args.DATA_NAME)
    args.load_path = os.path.join(args.working_path, 'checkpoints', args.NET_NAME, args.DATA_NAME, 'xxx.pth')

    return args
# args = {
#     'train_batch_size': 16,
#     'val_batch_size':32,
#     'lr': 0.001,
#     'epochs': 200,
#     'gpu': True,
#     'dev_id': 0,
#     'multi_gpu': None,  #"0,1,2,3",
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#     'print_freq': 50,
#     'predict_step': 5,
#     'img_size': 256,
#     'crop_size': 512,
#     'load_premodel':True,
#     'seed': 2137,
#     'pred_dir': os.path.join(working_path, 'results',  DATA_NAME),
#     'chkpt_dir': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME),
#     'chkpt_path': ' ',  # pretrain model path
#     'log_dir': os.path.join(working_path, 'logs', NET_NAME, DATA_NAME),
#     'load_path': os.path.join(working_path, 'checkpoints', NET_NAME, DATA_NAME, 'xxx.pth')}
# python train_loss3_detachlook2.py
########################## Parameters ########################


def set_seed(seed):
    """
    Set all random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU

    # Configure CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


# -------------------------------------------------
#  MOVE seed_worker HERE (outside of main)
# -------------------------------------------------
def seed_worker(worker_id):
    """
    Set seed for DataLoader workers
    """
    # args is global, so this is fine
    worker_seed = args['seed'] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class UncertaintyAwareLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=20.0, theta=0.5):
        super(UncertaintyAwareLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.theta = theta
        self.epsilon = 1e-5

    def normalize_entropy(self, entropy_map):
        max_val = entropy_map.max()
        if max_val > 0:
            return entropy_map / max_val
        return entropy_map

    def focal_loss_ua(self, inputs, targets, certainty):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** 2.0)  # Gamma=2.0
        weighted_loss = loss * certainty
        return weighted_loss.mean()

    def dice_loss_ua(self, inputs, targets, certainty):
        inputs = torch.sigmoid(inputs)
        mask = (certainty > self.theta).float()
        inputs_filtered = inputs * mask
        targets_filtered = targets * mask
        intersection = (inputs_filtered * targets_filtered).sum()
        union = inputs_filtered.sum() + targets_filtered.sum()
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice

    def forward(self, inputs, targets, entropy_map):
        u = self.normalize_entropy(entropy_map)
        c = 1.0 - u
        loss_focal = self.focal_loss_ua(inputs, targets, c)
        loss_dice = self.dice_loss_ua(inputs, targets, c)
        return loss_focal * self.focal_weight + loss_dice * self.dice_weight




def main():
    set_seed(args['seed'])
    args = parse_args()    
    if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
    if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
    if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
    writer = SummaryWriter(args['log_dir'])

    net = Net()
    # print(args['load_path'])
    # net.load_state_dict(torch.load(args['load_path']), strict=True)
    if args['multi_gpu']:
        net = torch.nn.DataParallel(net, [int(id) for id in args['multi_gpu'].split(',')])
    net.to(device=torch.device('cuda', int(args['dev_id'])))

    if args['load_premodel']:
        state_dict = torch.load(args["chkpt_path"], map_location="cpu", weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'segmenter2' not in k:
                if 'module.' in k:
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

        # Print loading status
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        print(f"✅ Successfully loaded {len(pretrained_dict)} parameters out of {len(model_dict)} total parameters.")
        if len(pretrained_dict) == 0:
            print("⚠️ Warning: No pre-trained parameters were loaded. Please check the model structure or pre-trained file.")
        else:
            print("✅ Pre-trained weights loaded:", list(pretrained_dict.keys())[:5])  # Print first 5 keys as example
        net.load_state_dict(pretrained_dict, strict=False)
        net.to(torch.device('cuda', int(args['dev_id']))).eval()

    train_set = RS.RS('train', random_crop=False, crop_nums=10, crop_size=args['crop_size'],
                      random_flip=True)  # '5_train_supervised',
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True,
                              worker_init_fn=seed_worker)
    val_set = RS.RS('test', sliding_crop=False, crop_size=args['crop_size'], random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False,
                            worker_init_fn=seed_worker)



    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_loader, net, optimizer, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, optimizer, val_loader):
    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sem = LatentSimilarity(T=3.0).to(torch.device('cuda', int(args['dev_id'])))
    criterion_ua = UncertaintyAwareLoss(theta=0.5).to(torch.device('cuda', int(args['dev_id'])))

    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args)
            imgs_A, imgs_B, labels, filenames = data
            if args['gpu']:
                imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
                imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
                labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

            optimizer.zero_grad()
            # outputs1, outputs2, outputs3, outA, outB, uncertainty_map = net(imgs_A, imgs_B)
            outputs1, outputs2, outputs3, outA, outB, uncertainty_map = net(
                imgs_A, imgs_B,
                filenames=filenames,
                save_visuals=False
            )
            # outputs = net(imgs_A, imgs_B)
            assert outputs1.shape[1] == 1
            loss_bn1 = F.binary_cross_entropy_with_logits(outputs1, labels)
            loss_bn2 = F.binary_cross_entropy_with_logits(outputs2, labels)
            loss_bn3 = F.binary_cross_entropy_with_logits(outputs3, labels)

            loss_ua = criterion_ua(outputs3, labels, uncertainty_map)

            loss_t = criterion_sem(outA, outB, labels)
            loss = loss_bn1 + loss_bn2 + loss_bn3 + loss_t + 0.005 * loss_ua
            # loss = loss_bn1 + loss_bn2 + loss_bn3 + loss_t + 1 * loss_ua
            # loss = loss_bn
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs3.cpu().detach()
            preds = F.sigmoid(outputs).numpy()
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))

        val_F, val_acc, val_IoU, val_loss, val_precision, val_recall = validate(val_loader, net, curr_epoch)
        if val_F > bestF:
            bestF = val_F
            bestacc = val_acc
            bestIoU = val_IoU
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME + '_F%.2f_P%.2f_R%.2f.pth' % (
            val_F * 100, val_precision * 100, val_recall * 100 )))
        if val_F < bestF:
            # Obtain the underlying model for access save_dir
            _net = net.module if isinstance(net, torch.nn.DataParallel) else net
            if hasattr(_net, 'save_dir'):
                dir_to_remove = os.path.join(_net.save_dir, DATA_NAME, f'epoch_{curr_epoch}')
                if os.path.exists(dir_to_remove):
                    shutil.rmtree(dir_to_remove)
                    print(f'[Info] Epoch {curr_epoch} visuals removed (F1 {val_F:.4f} < Best {bestF:.4f})')

            
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1 score: %.2f IoU %.2f' \
              % (curr_epoch, args['epochs'], time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
                 bestIoU * 100))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, curr_epoch):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()

    preds = []
    GTs = []

    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels, filenames = data

        if args['gpu']:
            imgs_A = imgs_A.to(torch.device('cuda', int(args['dev_id']))).float()
            imgs_B = imgs_B.to(torch.device('cuda', int(args['dev_id']))).float()
            labels = labels.to(torch.device('cuda', int(args['dev_id']))).float().unsqueeze(1)

        with torch.no_grad():
            output1, output2, output3, _, _, _ = net(
                imgs_A, imgs_B,
                filenames=filenames,
                epoch=curr_epoch,
                dataset_name=DATA_NAME, 
                save_visuals=False, 
                # save_visuals=True, 
                labels=labels
            )
            loss_bn1 = F.binary_cross_entropy_with_logits(output1, labels)
            loss_bn2 = F.binary_cross_entropy_with_logits(output2, labels)
            loss_bn3 = F.binary_cross_entropy_with_logits(output3, labels)

            loss = loss_bn1 + loss_bn2 + loss_bn3

            output3 = F.sigmoid(output3)

        val_loss.update(loss.cpu().detach().numpy())

        outputs3 = output3.cpu().detach().numpy()

        labels = labels.cpu().detach().numpy()
        for (pred3, label) in zip(outputs3, labels):

            outputs3 = pred3.squeeze() > 0.5
            outputs3 = outputs3.astype(np.int64)
            preds.append(outputs3)
            label = label.astype(np.int64)
            GTs.append(label)


    score_dict = get_mIoU(2, GTs, preds)
    val_F, val_acc, val_IoU, val_precision, val_recall = score_dict['F1_1'], score_dict['acc'], score_dict['iou_1'], score_dict['precision_1'], score_dict['recall_1']
    curr_time = time.time() - start
    print('%.1fs Val loss %.2f Acc %.2f F %.2f P %.2f R %.2f' % (
    curr_time, val_loss.average(), val_acc * 100, val_F * 100, val_precision * 100, val_recall * 100))
    return val_F, val_acc, val_IoU, val_loss.avg, val_precision, val_recall

def adjust_lr(optimizer, curr_iter, all_iter, args):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** 3.0)
    running_lr = args['lr'] * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    args = parse_args()
    main()
