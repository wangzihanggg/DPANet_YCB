#!/usr/bin/python3 #coding=utf-8

import os
import sys
import datetime
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tensorboardX import SummaryWriter

    BOARD_FLAG = True
except:
    BOARD_FLAG = False

# from lib import dataset
# from lib.dataset import train_collate_fn
# from lib import my_dataset
# from lib.my_dataset import train_collate_fn
from lib import my_dataset_1
from lib.my_dataset_1 import train_collate_fn
from network import Segment
import logging as logger
# from lib.data_prefetcher import DataPrefetcher
from lib.my_data_prefetcher import DataPrefetcher
import argparse
from lib.utils import load_model
from PIL import Image

DATA_PATH = "/project/1_2301/DPANet-master/YCB_Video_Dataset/data"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(conf, Dataset, Network):
    # dataset
    cfg = Dataset.Config(datapath=DATA_PATH, savepath=conf.savepath, mode='train', batch=conf.bz_size, lr=conf.lr,
                         momen=0.9, decay=conf.decay, epoch=conf.epochs, train_scales=[224, 256, 320])
    # data = Dataset.RGBDData(cfg)
    data = Dataset.YCBData(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8, drop_last=True,
                        collate_fn=train_collate_fn)
    # network

    # net = Network(backbone='resnet50', cfg=cfg, norm_layer=nn.SyncBatchNorm)
    net = Network(backbone='resnet50', cfg=cfg)

    #
    if os.path.exists(conf.model_pt):
        msg = "Loading pretrained model_pt:%s" % conf.model_pt
        print(msg)
        logger.info(msg)
        net.load_state_dict(torch.load(conf.model_pt), strict=True)

    net.train()
    # net = nn.DataParallel(net)
    net.cuda()
    ## parameter
    rgb_base, rgbd_base, d_base, head = [], [], [], []
    fc_params = []
    for name, param in net.named_parameters():
        if 'bkbone_rgbd' in name:
            rgbd_base.append(param)
        elif 'bkbone_rgb' in name:
            rgb_base.append(param)
        elif 'bkbone_d' in name:
            d_base.append(param)
        elif 'fc' in name:
            print("add fc_params:", name)
            fc_params.append(param)
        else:
            head.append(param)
    assert len(rgbd_base) == 0
    optimizer = torch.optim.SGD([{'params': rgb_base}, {'params': d_base}, {'params': fc_params}, {'params': head}],
                                lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay)  # nesterov=True)
    if BOARD_FLAG:
        sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(cfg.epoch):
        exist_img_names = []
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[2]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        optimizer.param_groups[3]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        step = -1
        prefetcher = DataPrefetcher(loader, cnt=3)
        # image, depth, mask, gate_gt = prefetcher.next()
        image, depth, mask, img_name = prefetcher.next()
        while image is not None:
            out, out2_1, out3_1, out4_1, out5_1, \
            out2_2, out3_2, out4_2, out5_2, gate = net.forward(image, depth)
            # dominant loss
            new_out = out.detach().cpu().numpy()
            new_mask = mask.detach().cpu().numpy()
            # new_out = out.detach().cpu().numpy() * 255
            # new_mask = mask.detach().cpu().numpy() * 255
            for i in range(new_out.shape[0]):
                if img_name[i] in exist_img_names:
                    continue
                else:
                    new_out[i][0] = (new_out[i][0] - new_out[i][0].min()) \
                                    / (new_out[i][0].max() - new_out[i][0].min())
                    new_out[i][0] = new_out[i][0] * 21  # 标签只有21类
                    new_out[i][0] = new_out[i][0].astype(np.uint8)
                    img = Image.fromarray(new_out[i][0])
                    img.convert('L').save(img_name[i][:-10] + "-pred_mask.png")
                    print("1", new_out[i][0].min(), new_out[i][0].max(), img_name[i])
                    exist_img_names.append(img_name[i])
                    # print("2", new_mask.min(), new_mask.max())
            dom_loss = F.binary_cross_entropy_with_logits(out, mask)
            # aux. loss
            loss2_1 = F.binary_cross_entropy_with_logits(out2_1, mask)
            loss3_1 = F.binary_cross_entropy_with_logits(out3_1, mask)
            loss4_1 = F.binary_cross_entropy_with_logits(out4_1, mask)
            loss5_1 = F.binary_cross_entropy_with_logits(out5_1, mask)
            loss2_2 = F.binary_cross_entropy_with_logits(out2_2, mask)
            loss3_2 = F.binary_cross_entropy_with_logits(out3_2, mask)
            loss4_2 = F.binary_cross_entropy_with_logits(out4_2, mask)
            loss5_2 = F.binary_cross_entropy_with_logits(out5_2, mask)
            # regression
            # reg_loss = F.smooth_l1_loss(gate, gate_gt) * 2
            # loss = dom_loss + reg_loss + 0.8 * (loss2_1 * 1 + loss3_1 * 0.8 + loss4_1 * 0.6 + loss5_1 * 0.4) + 0.8 * (
            #             loss2_2 * 1 + loss3_2 * 0.8 + loss4_2 * 0.6 + loss5_2 * 0.4)

            loss = dom_loss + 0.8 * (loss2_1 * 1 + loss3_1 * 0.8 + loss4_1 * 0.6 + loss5_1 * 0.4) + 0.8 * (
                    loss2_2 * 1 + loss3_2 * 0.8 + loss4_2 * 0.6 + loss5_2 * 0.4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            global_step += 1
            step += 1
            if BOARD_FLAG:
                sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                sw.add_scalars('loss',
                               {'dom_loss': dom_loss.item(), 'loss2_1': loss2_1.item(), 'loss3_1': loss3_1.item(),
                                'loss4_1': loss4_1.item(), 'loss5_1': loss5_1.item(),
                                'loss': loss.item()}, global_step=global_step)
                # sw.add_scalars('loss',
                #                {'dom_loss': dom_loss.item(), 'loss2_1': loss2_1.item(), 'loss3_1': loss3_1.item(),
                #                 'loss4_1': loss4_1.item(), 'loss5_1': loss5_1.item(),
                #                 'reg_loss': reg_loss.item(), 'loss': loss.item()}, global_step=global_step)
            if step % 10 == 0:
                # msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | dom_loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss2_2=%.6f | loss3_2=%.6f | loss4_2=%.6f | loss5_2=%.6f | reg_loss=%.6f' % (
                # datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                # loss.item(), dom_loss.item(), loss2_1.item(), loss3_1.item(), loss4_1.item(), loss5_1.item(),
                # loss2_2.item(), loss3_2.item(), loss4_2.item(), loss5_2.item(), reg_loss.item())
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | dom_loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss2_2=%.6f | loss3_2=%.6f | loss4_2=%.6f | loss5_2=%.6f' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss.item(), dom_loss.item(), loss2_1.item(), loss3_1.item(), loss4_1.item(), loss5_1.item(),
                    loss2_2.item(), loss3_2.item(), loss4_2.item(), loss5_2.item())
                print(msg)
                logger.info(msg)
            # image, depth, mask, gate_gt = prefetcher.next()
            image, depth, mask, img_name = prefetcher.next()

        if (epoch + 1) in [cfg.epoch, cfg.epoch - 1, cfg.epoch - 2, \
                           cfg.epoch - 3, cfg.epoch - 4]:  # or dom_loss.item() <= 0.025:
            # logger.info("saving model-%s ..., loss:%s"%(epoch+1, dom_loss.item()))
            # torch.save(net.module.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + ".pth")


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description="train model")
    conf.add_argument("--tag", type=str, default="res50")
    conf.add_argument("--savepath", type=str, help="where to save models?", default="res50_res")
    conf.add_argument("--lr", type=float, default=0.05)
    # conf.add_argument("--bz_size", type=int, default=32)
    conf.add_argument("--bz_size", type=int, default=2)
    # conf.add_argument("--epochs", type=int, default=30)
    # conf.add_argument("--epochs", type=int, default=2)
    conf.add_argument("--epochs", type=int, default=50)
    conf.add_argument("--decay", type=float, default=1e-4)
    conf.add_argument("--seed", type=int, default=1997)
    # conf.add_argument("--gpu", type=str, default="0,1,2,3")
    conf.add_argument("--gpu", type=str, default="0")
    conf.add_argument("--model_pt", type=str, default="None")

    args = conf.parse_args()
    logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S', \
                       filename="train_%s.log" % (args.tag), filemode="w")
    logger.info("Configuration:{}".format(args))
    logger.info("SEED:%s, gpu:%s" % (args.seed, args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(args.seed)
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    # train(args, my_dataset, Segment)
    train(args, my_dataset_1, Segment)
