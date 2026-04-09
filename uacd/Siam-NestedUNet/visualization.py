'''
This file is used to save the output image
'''

import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, initialize_metrics
import os
from tqdm import tqdm
import cv2
import numpy as np

if not os.path.exists('./output_img/S2Looking/base'):
    os.mkdir('./output_img/S2Looking/base')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt, batch_size=1)

path = 'path/of/model'   # the path of the model
checkpoint = torch.load(path)

if isinstance(checkpoint, torch.nn.Module):
    model = checkpoint
else:
    from utils.helpers import load_model
    model = load_model(opt, dev)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
model.eval()
index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels, filename in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
        cd_preds = model(batch_img1, batch_img2)
        batch_size = len(batch_img1)
        cd_preds = cd_preds[0]
        _, cd_preds = torch.max(cd_preds, 1)
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        filename = ''.join(filename)

        base_name = os.path.splitext(filename)[0]

        file_path = './output_img/WHU-CD/base/' + base_name + '.png'
        cv2.imwrite(file_path, cd_preds)
        print(f"Saved: {file_path}")
        index_img += 1
