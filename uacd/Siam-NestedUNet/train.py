import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import os
import logging
import json
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
save_path = 'save/path'
class UncertaintyAwareLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=20.0, theta=0.5):
        """
        Loss function based on ununcertainty -aware Fine-tuning in the paper
        theta: Deterministic threshold. Only pixels with a determinism greater than theta participate in the Dice Loss calculation
        """
        super(UncertaintyAwareLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.theta = theta
        self.epsilon = 1e-5

    def normalize_entropy(self, entropy_map):
        # Normalize the entropy to [0, 1]
        max_val = entropy_map.max()
        if max_val > 0:
            return entropy_map / max_val
        return entropy_map

    def focal_loss_ua(self, inputs, targets, certainty):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** 2.0)  # Gamma=2.0

        # Core: Based on deterministic weighting (Certainty Weighting)
        weighted_loss = loss * certainty
        return weighted_loss.mean()

    def dice_loss_ua(self, inputs, targets, certainty):
        inputs = torch.sigmoid(inputs)

        # Core: Deterministic threshold filtering (Thresholding)
        mask = (certainty > self.theta).float()

        inputs_filtered = inputs * mask
        targets_filtered = targets * mask

        intersection = (inputs_filtered * targets_filtered).sum()
        union = inputs_filtered.sum() + targets_filtered.sum()

        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice

    def forward(self, inputs, targets, entropy_map):
        # 1. Normalized entropy yields uncertainty u
        u = self.normalize_entropy(entropy_map)
        # 2. Computational certainty c = 1 - u
        c = 1.0 - u
        # c = c.detach()  # Gradient blocking, not backpropagating to the generation process of the Uncertainty Map

        loss_focal = self.focal_loss_ua(inputs, targets, c)
        loss_dice = self.dice_loss_ua(inputs, targets, c)

        return loss_focal * self.focal_weight + loss_dice * self.dice_weight
"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)


train_loader, val_loader = get_loaders(opt)

"""
Load Model then define other aspects of the model
"""
logging.info('LOADING Model')

model = load_model(opt, dev)
pretrained_path = 'pretrained/model/path' 

if pretrained_path and os.path.exists(pretrained_path):
    logging.info(f'Loading pretrained weights from: {pretrained_path}')
    try:
        checkpoint = torch.load(pretrained_path, map_location=dev, weights_only=False)
        # Handle the situation where the checkpoint is the entire model or state_dict
        if isinstance(checkpoint, nn.Module):
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logging.info('Pretrained weights loaded successfully!')
    except Exception as e:
        logging.error(f'Error loading pretrained weights: {e}')
else:
    logging.info('No pretrained path specified or file not found. Training from scratch.')
criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

"""
 Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1
criterion_ua = UncertaintyAwareLoss(theta=0.5).to(torch.device('cuda', 0))
for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1
        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop
        cd_preds, uncertainty_map = model(batch_img1, batch_img2)

        loss_input = [cd_preds]
        final_pred_tensor = cd_preds

        cd_loss = criterion(loss_input, labels)

        cd_preds1 = final_pred_tensor[:, 1:2, :, :]
            
        labels1 = labels.unsqueeze(1).float()

        loss_ua = criterion_ua(cd_preds1, labels1, uncertainty_map)
        
        loss = cd_loss + 0.005 * loss_ua
        loss.backward()
        optimizer.step()

        # cd_preds = [cd_preds]
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               zero_division=0,
                               pos_label=1)

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            cd_preds, uncertainty_map = model(batch_img1, batch_img2)

            loss_input = [cd_preds] 
            final_pred_tensor = cd_preds

            cd_loss = criterion(loss_input, labels)

            cd_preds1 = final_pred_tensor[:, 1:2, :, :]
            
            labels1 = labels.unsqueeze(1).float()

            loss_ua = criterion_ua(cd_preds1, labels1, uncertainty_map)
        
            loss = cd_loss + 0.005 * loss_ua

            # cd_preds = [cd_preds]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
            cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 zero_division=0,
                                 pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        """
        Store the weights of good epochs based on validation results
        """
        # if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
        #         or
        #         (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
        #         or
        #         (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):
        if mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']:

            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')
            metadata['validation_metrics'] = mean_val_metrics

            # Save model and log
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path + '/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model, save_path + '/checkpoint_epoch_'+str(epoch)+'.pt') 

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics


        print('An epoch finished.')
writer.close()  # close tensor board
print('Done!')
