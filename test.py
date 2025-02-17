
from scipy.ndimage import gaussian_filter
from funcs import *
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter
from funcs import denormalization
from mvtec import MVTecDataset
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from numpy import ndarray
import pandas as pd
from skimage import measure
from statistics import mean
import os
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def main(object, path):
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--obj', type=str, default=object)
    parser.add_argument('--data_type', type=str, default='Mvtec')
    parser.add_argument('--data_path', type=str, default='D:/chenku/anomaly_dection/mvT')
    parser.add_argument('--checkpoint_dir', type=str, default=path)
    parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=3457)


    args = parser.parse_args()

    # load model and dataset
    args.input_channel = 1 if args.grayscale else 3

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    checkpoint = torch.load(args.checkpoint_dir)
    bn.load_state_dict(checkpoint['bn'])
    decoder.load_state_dict(checkpoint['decoder'])

    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed) + '/AM'
    if not os.path.exists(args.save_dir):
         os.makedirs(args.save_dir)


    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    scores, test_imgs, recon_imgs, gt_list, gt_mask_list, image_scores, pro = test(bn, decoder, test_loader, encoder)
    scores = np.asarray(scores)
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    img_scores = np.asarray(image_scores)
    gt_list = np.asarray(gt_list)

    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc * 100))

    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc * 100))
    ap_rocauc = np.mean(pro)
    print('ap ROCAUC: %.3f' % (ap_rocauc * 100))
    print_log(('image: {:.3f} pixel:{:.3f} PRO:{:.3f}'.format(img_roc_auc * 100, per_pixel_rocauc * 100, ap_rocauc * 100)),log)


def get_anomap(output, Dn, data):
    n, c, h, w = data.shape
    anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(output[0], Dn[0])
    anomaly_map1_kd = anomaly_map1_kd.unsqueeze(0)
    anomaly_map1_kdx = F.interpolate(anomaly_map1_kd, size=(h, w), mode='bilinear', align_corners=True)

    anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(output[1], Dn[1])
    anomaly_map2_kd = anomaly_map2_kd.unsqueeze(0)
    anomaly_map2_kdx = F.interpolate(anomaly_map2_kd, size=(h, w), mode='bilinear', align_corners=True)
    # \
    anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(output[2], Dn[2])
    anomaly_map3_kd = anomaly_map3_kd.unsqueeze(0)
    anomaly_map3_kdx = F.interpolate(anomaly_map3_kd, size=(h, w), mode='bilinear', align_corners=True)  # \

    max_value1 = anomaly_map1_kd.max()
    max_value2 = anomaly_map2_kd.max()
    max_value3 = anomaly_map3_kd.max()

    variance1 = torch.var(anomaly_map1_kdx)
    variance2 = torch.var(anomaly_map2_kdx)
    variance3 = torch.var(anomaly_map3_kdx)

    return [max_value1, max_value2, max_value3], [variance1, variance2, variance3], anomaly_map1_kdx, anomaly_map2_kdx, anomaly_map3_kdx



def test(bn, decoder, test_loader, encoder):
    bn.eval()
    decoder.eval()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    image_scores = []
    aupro_list = []
    for (data, _, label, mask) in tqdm(test_loader):
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy().astype(int))
        gt_mask_list.extend(mask.cpu().numpy().astype(int).ravel())
        with torch.no_grad():
            data = data.to(device)
            output = encoder(data)
            Dn = decoder(bn(output))
            value, std, am1, am2, am3 = get_anomap(output, Dn, data)
            combined = list(zip(value, std))
            sorted_combined = sorted(combined, key=lambda x: x[1])
            values_of_min_std = [val for val, std in sorted_combined[:1]]
            value_of_min_std = values_of_min_std[0]
            anomaly_map = (am1 + am2 + am3) / 3
            final_S = (1 + value_of_min_std) * (am1 + am2 + am3) / 3


        score = anomaly_map.squeeze(0).cpu().numpy()
        score = gaussian_filter(score, sigma=4)
        img_score = final_S.squeeze(0).cpu().numpy()
        img_score = gaussian_filter(img_score, sigma=4)
        s = np.max(img_score)
        scores.append(score)
        recon_imgs.extend(data.cpu().numpy())
        image_scores.append(s)
        if label.item() != 0:
            aupro_list.append(compute_pro(mask.squeeze(0).cpu().numpy().astype(int), score))
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list, image_scores, aupro_list

if __name__ == '__main__':
    main('tile', './mvtec/tile/seed_/XXXXXX.pth')