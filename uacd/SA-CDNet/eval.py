
import os
import time
from utils.metric_tool import get_mIoU
import cv2
import numpy as np
import torch.autograd
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
working_path = os.path.abspath('.')

from collections import OrderedDict
from tqdm import tqdm
# save_visuals It can be used to switch on and off and save the visualization
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###################### Data and Model ########################
from models.SAM_Fusion4 import SAM_CD as Net
NET_NAME = ''
from datasets import data as RS
DATA_NAME = ''
########################## Parameters ########################
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script for change detection')
    
    parser.add_argument('--NET_NAME', type=str, default='YOUR_NET_NAME', help='Network name')
    parser.add_argument('--DATA_NAME', type=str, default='YOUR_DATASET_NAME', help='Dataset name')
    parser.add_argument('--dataset_type', type=str, default='test', choices=['train', 'val', 'test'], help='Dataset split to use')
    
    parser.add_argument('--working_path', type=str, default=os.path.abspath('.'), help='Working directory')
    parser.add_argument('--chkpt_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--pred_dir', type=str, default=None, help='Directory to save predictions')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for logs')
    
    parser.add_argument('--val_batch_size', type=int, default=32, help='Validation batch size')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop size for images')
    parser.add_argument('--num_workers', type=int, default=64, help='Number of data loader workers')
    parser.add_argument('--predict_step', type=int, default=5, help='Prediction step')
    
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false', help='Do not use GPU')
    parser.add_argument('--dev_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs')
    parser.add_argument('--multi_gpu', action='store_true', default=False, help='Use multiple GPUs')
    
    parser.add_argument('--save_visuals', action='store_true', default=True, help='Save visualization results')
    parser.add_argument('--no-save_visuals', dest='save_visuals', action='store_false', help='Do not save visualization')
    
    parser.add_argument('--sliding_crop', action='store_true', default=False, help='Use sliding crop')
    parser.add_argument('--random_flip', action='store_true', default=False, help='Use random flip')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save evaluation results to file')
    
    return parser.parse_args()
# args = {
#     'train_batch_size': 16,
#     'val_batch_size':32,
#     'lr': 0.01,
#     'epochs': 200,
#     'gpu': True,
#     'dev_id': 0,
#     'multi_gpu': None,  #"0,1,2,3",
#     'weight_decay': 5e-4,
#     'momentum': 0.9,
#     'print_freq': 50,
#     'predict_step': 5,
#     'crop_size': 512,
#     'chkpt_path':os.path.join(working_path, 'ckp/SA-CDNet/main/LEVIR+/ckp.pth'),
#     'pred_dir': os.path.join(working_path, 'results', DATA_NAME),}

# if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
# writer = SummaryWriter(args['log_dir'])
def main():
    global NET_NAME, DATA_NAME
    args = parse_args()
    NET_NAME = args.NET_NAME
    DATA_NAME = os.path.join(args.NET_NAME, args.DATA_NAME)
    net = Net()
    # net.load_state_dict(torch.load(args['load_path']), strict=False)
    net.to(device=torch.device('cuda', int(args['dev_id'])))
    val_set = RS.RS('test', sliding_crop=False, crop_size=args['crop_size'], random_flip=False)
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=64, shuffle=False)
    state_dict = torch.load(args["chkpt_path"], map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict, strict=True)
    net.to(torch.device('cuda', int(args['dev_id']))).eval()
    validate(val_loader,net)


def validate(val_loader, net):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()

    preds = []
    GTs = []
    # for data in tqdm(val_loader):
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
                dataset_name=DATA_NAME,  # Use the global variable DATA_NAME
                # save_visuals=False, 
                save_visuals=True, 
                labels=labels
            )
            output3 = F.sigmoid(output3)

        outputs3 = output3.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for (pred3, label) in zip(outputs3, labels):
            outputs3 = pred3.squeeze() > 0.5
            outputs3 = outputs3.astype(np.int64)

            preds.append(outputs3)
            label = label.astype(np.int64)
            GTs.append(label)
    score_dict = get_mIoU(2, GTs, preds)
    F1, acc, IoU, precision, recall = score_dict['F1_1'], score_dict['acc'], score_dict['iou_1'], score_dict['precision_1'], score_dict['recall_1']
    print('ACC: ' + str(acc))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(F1))
    print('IoU: ' + str(IoU))
    log_file_path = os.path.join(args['pred_dir'], 'val_results.txt')

    # Obtain the current time to facilitate the distinction of multiple operation records
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Open it in 'a' (append) mode so that the new result will be appended to the end of the file instead of being overwritten
    with open(log_file_path, 'a') as f:
        f.write(f"========================================\n")
        f.write(f"Time: {current_time}\n")
        f.write(f"Checkpoint: {os.path.basename(args['chkpt_path'])}\n")
        f.write(f"ACC: {acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {F1}\n")
        f.write(f"IoU: {IoU}\n")
        f.write(f"========================================\n\n")

    print(f"The verification result has been saved to: {log_file_path}")


if __name__ == '__main__':
    main()