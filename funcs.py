import numpy as np
import torch
from utils import print_log
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import torch.nn as nn
from numpy import ndarray
import pandas as pd
from skimage import measure
from statistics import mean
from sklearn.metrics import auc

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    renormalized_tensor = x.clone()
    for i in range(len(mean)):
        renormalized_tensor[:, i, :, :] = renormalized_tensor[:, i, :, :] * std[i] + mean[i]
    return renormalized_tensor


def norm(x):
    """Convert the range from [0, 1] to [-1, 1]."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize_transform = transforms.Normalize(mean, std)
    normalized_tensor = normalize_transform(x)
    return normalized_tensor

def cut(img, t, b):
    # h, w, c = img.shape
    x = np.random.randint(0, img.shape[1] - t)
    y = np.random.randint(0, img.shape[0] - b)
    if (x - t) % 2 == 1:
        t -= 1
    if (y - b) % 2 == 1:
        b -= 1

    roi = img[y:y + b, x:x + t]
    return roi


def paste_patch(img, patch):
    imgh, imgw, imgc = img.shape
    patchh, patchw, patchc = patch.shape

    patch_h_position = random.randrange(1, round(imgh) - round(patchh) - 1)
    patch_w_position = random.randrange(1, round(imgw) - round(patchw) - 1)
    pasteimg = np.copy(img)
    pasteimg[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw, :] = patch + 0.2 * img[patch_h_position:patch_h_position + patchh,
    patch_w_position:patch_w_position + patchw, :]

    return pasteimg


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos = nn.functional.cosine_similarity(feature1, feature2, dim=1)
        ano_map = torch.ones_like(cos) - cos
        loss = (ano_map.view(ano_map.shape[0], -1).mean(-1)).mean()
        return loss

class loss_fucntion(nn.Module):
    def __init__(self):
        super(loss_fucntion, self).__init__()

    def forward(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))

        loss = loss / (len(a))
        return loss


class EarlyStop():
    """Used to early stop the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=True, delta=0, save_name="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, log):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print_log((f'EarlyStopping counter: {self.counter} out of {self.patience}'), log)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, log)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, bn, decoder, log):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_log((f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'),
                      log)
        state = {'bn': bn.state_dict(), 'decoder': decoder.state_dict()}
        torch.save(state, self.save_name)
        self.val_loss_min = val_loss



def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool_)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


def psedu_generate(image_np, t):
    transform = transforms.Compose([
        Normalize(),
        ToTensor(),
    ])
    rotated_list = []
    for i in range(image_np.size(0)):
        np_img = image_np[i]
        patch_img = cut(np_img, t, t)
        patch_img = paste_patch(np_img, patch_img)
        img_noise = transform(patch_img)
        img_noise = torch.unsqueeze(img_noise, dim=0)
        rotated_list.append(img_noise)
    img_noise = torch.cat(rotated_list, dim=0)
    return img_noise


