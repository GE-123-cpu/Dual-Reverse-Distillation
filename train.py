import argparse
import time
import torch.optim as optim
from tqdm import tqdm
from mvtec import *
from funcs import *
from utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
import random
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from test import test


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main(object):
    parser = argparse.ArgumentParser(description='anomaly detection')
    parser.add_argument('--obj', type=str, default=object)
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='D:/chenku/anomaly_dection/mvT')
    parser.add_argument('--epochs', type=int, default=200, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './' + args.data_type + '/' + args.obj + '/seed_{}/'.format(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}  # {args.obj}
    print_log(state, log)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=args.lr,
                           weight_decay=args.weight_decay, betas=(0.5, 0.999))

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    save_name = os.path.join(args.save_dir, '{}_{}_model.pth'.format(args.obj, args.prefix))
    early_stop = EarlyStop(patience=30, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    test_interval = 4
    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(args, optimizer, epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train(bn, decoder, epoch, train_loader, optimizer, log, encoder)

        if (epoch + 1) % test_interval == 0:
           scores, test_imgs, recon_imgs, gt_list, gt_mask_list, i_scores = test_(bn, decoder, test_loader, encoder)
           scores = np.asarray(scores)
           max_anomaly_score = scores.max()
           min_anomaly_score = scores.min()
           scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

           img_scores = i_scores
           gt_list = np.asarray(gt_list)
           fpr, tpr, _ = roc_curve(gt_list, img_scores)
           img_roc_auc = roc_auc_score(gt_list, img_scores)
           gt_mask = np.asarray(gt_mask_list)
           per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
           best_score = - (per_pixel_rocauc + img_roc_auc) * 100
           print_log(('epoch: {} image: {:.3f} pixel: {:.3f}'.format(epoch, img_roc_auc * 100, per_pixel_rocauc * 100)), log)

           if (early_stop(best_score, bn, decoder, log)):
              break

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()



def train(bn, decoder, epoch, train_loader, optimizer, log, encoder):
    bn.train()
    decoder.train()
    seg_losses = AverageMeter()
    kd_losses = AverageMeter()
    rec_losses = AverageMeter()
    losses = AverageMeter()
    L1 = nn.L1Loss(reduction='mean')
    cos_batch = CosineLoss()
    cos_single = loss_fucntion()

    for (data, img_np, _, _) in tqdm(train_loader):
        optimizer.zero_grad()
        n, c, h, w = data.shape

        large_value = random.randint(150, 200)
        normal_value = random.randint(64, 100)
        small_value = random.randint(10, 32)

        weights = [0.2, 0.3, 0.5]  # 选择large的概率为0.2，normal的概率为0.3，small的概率为0.5
        values = [large_value, normal_value, small_value]

        # 根据概率选择一个值
        t = random.choices(values, weights, k=1)[0]

        img_noise_m = psedu_generate(img_np, t)


        data = data.to(device)
        noise = img_noise_m.to(device)
        mask = torch.ones(1, h, w).to(device) - F.cosine_similarity(noise, data)
        mask = mask.unsqueeze(dim=1)

        output = encoder(data)
        output_noise = encoder(noise)

        ########
        output_noise1 = output_noise
        output_noise1[0] = output_noise[0] + torch.randn_like(output_noise[0]) * 0.1
        output_noise1[1] = output_noise[1] + torch.randn_like(output_noise[1]) * 0.1
        output_noise1[2] = output_noise[2] + torch.randn_like(output_noise[2]) * 0.1
        ########
        p = bn(output)
        pa = bn(output_noise1)
        f = decoder(p)
        fa = decoder(pa)


        anomaly_map1_kd = torch.ones(1, 64, 64).to(device) - F.cosine_similarity(output_noise[0], fa[0])
        anomaly_map1_kd = anomaly_map1_kd.unsqueeze(dim=1)
        anomaly_map1_kd = F.interpolate(anomaly_map1_kd, size=(h, w), mode='bilinear', align_corners=True)
        anomaly_map2_kd = torch.ones(1, 32, 32).to(device) - F.cosine_similarity(output_noise[1], fa[1])
        anomaly_map2_kd = anomaly_map2_kd.unsqueeze(dim=1)
        anomaly_map2_kd = F.interpolate(anomaly_map2_kd, size=(h, w), mode='bilinear', align_corners=True)
        anomaly_map3_kd = torch.ones(1, 16, 16).to(device) - F.cosine_similarity(output_noise[2], fa[2])
        anomaly_map3_kd = anomaly_map3_kd.unsqueeze(dim=1)
        anomaly_map3_kd = F.interpolate(anomaly_map3_kd, size=(h, w), mode='bilinear', align_corners=True)
        anomaly_map = (anomaly_map1_kd + anomaly_map2_kd + anomaly_map3_kd) / 3


        kd_loss = cos_batch(output[0], f[0]) + cos_batch(output[1], f[1]) + cos_batch(output[2], f[2]) +\
                  cos_single(output[0], f[0]) + cos_single(output[1], f[1]) + cos_single(output[2], f[2])

        seg_loss = 0.1 * L1(anomaly_map, mask)

        rec_loss = cos_single(output[0], fa[0]) + cos_single(output[1], fa[1]) + cos_single(output[2], fa[2])

        loss = rec_loss + kd_loss + seg_loss

        seg_losses.update(seg_loss.item(), data.size(0))
        kd_losses.update(kd_loss.item(), data.size(0))
        rec_losses.update(rec_loss.item(), data.size(0))
        losses.update(loss.item(), data.size(0))

        loss.backward()
        optimizer.step()

    print_log(('Train Epoch: {} seg_Loss: {:.6f} cos_Loss: {:.6f} rec_loss:{:.6f} all_Loss: {:.6f}'.format(epoch,
               seg_losses.avg, kd_losses.avg, rec_losses.avg, losses.avg)), log)


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
    #Max = (max_value1 + max_value2 + max_value3) / 3

    return [max_value1, max_value2, max_value3], [variance1, variance2, variance3], anomaly_map1_kdx, anomaly_map2_kdx, anomaly_map3_kdx



def test_(bn, decoder, test_loader, encoder):
    bn.eval()
    decoder.eval()
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    recon_imgs = []
    image_scores = []
    for (data, _, label, mask) in tqdm(test_loader):
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        test_imgs.extend(data.cpu().numpy())
        gt_list.extend(label.cpu().numpy().astype(int))
        gt_mask_list.extend(mask.cpu().numpy().astype(int).ravel())
        with torch.no_grad():
            #n, c, h, w = data.shape
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
    return scores, test_imgs, recon_imgs, gt_list, gt_mask_list, image_scores

if __name__ == '__main__':
    item_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper']

    for i in item_list:
        main(i)
