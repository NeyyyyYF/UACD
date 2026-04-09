import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in trainuloss.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = 'path/of/model'   # the path of the model
# model = torch.load(path, weights_only=False)
model = torch.load(path)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()


with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:
        
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[0]
        _, cd_preds = torch.max(cd_preds, 1)

        # Convert the tensor to a numpy array and flatten it
        labels_np = labels.data.cpu().numpy().flatten()
        preds_np = cd_preds.data.cpu().numpy().flatten()
        
        # Calculate the confusion matrix
        cm = confusion_matrix(labels_np, preds_np)
        
        # Process according to the shape of the confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            # There is only one category
            if labels_np.sum() == 0 and preds_np.sum() == 0:  # All are negative classes
                tn = cm[0, 0]
                fp, fn, tp = 0, 0, 0
            elif labels_np.sum() > 0 and preds_np.sum() > 0:  # All are positive classes
                tp = cm[0, 0]
                tn, fp, fn = 0, 0, 0
            else: 
                if labels_np.sum() > 0:
                    fn = cm[0, 0] 
                    tn, fp, tp = 0, 0, 0
                else:
                    fp = cm[0, 0] 
                    tn, fn, tp = 0, 0, 0
        else:
            raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
print(path)
