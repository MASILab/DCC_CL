from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn.functional as F
from utils import TwoCropTransform, AverageMeter
from utils import adjust_learning_rate, warmup_learning_rate
from utils import set_optimizer, save_model
from networks.resnet_big_6channels import SupConResNet, SupCEResNet, DenseResNet
from losses_intensity_weight import DCCLoss
from Data_loader_patch import Data_loader
import tqdm
import glob
import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

os.environ["CUDA_VISIBLE_DEVICES"]='0'


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagevu',
                        choices=['cifar10', 'cifar100', 'imagevu'], help='dataset')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', 'CrossEntropy', 'DenseCon'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.3,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = '/nfs/masi/leeh43/contrastive_learning/2D_data'
    opt.model_path = './save/SupCon/{}_temp_0.07_models_128x128_pv_nc_resnet50_lr_0.0005_bz_4_6channel_image_weight_T_0.3'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_0.07_tensorboard_128x128_pv_nc_resnet50_lr_0.0005_bz_4_6channel_image_weight_T_0.3'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    CLASSES = [
      'bk', 'spleen', 'right_kidney', 'left_kidney', 'gall_bladder', 'esophagus', 'liver', 'stomach',
      'aorta', 'IVC', 'PSV', 'pancreas', 'RAD']
    if opt.dataset == 'imagevu':
        nc_root = os.path.join('/nfs/masi/leeh43/Supcon_learning/Organ_patches_2D_NC_2')
        pv_root = os.path.join('/nfs/masi/leeh43/Supcon_learning/Organ_patches_2D_PV_2')

        nc_label_train = []
        nc_prior_train = []
        nc_img_train = []
        nc_label_valid = []
        nc_prior_valid = []
        nc_img_valid = []
        pv_label_train = []
        pv_prior_train = []
        pv_img_train = []
        pv_label_valid = []
        pv_prior_valid = []
        pv_img_valid = []
        for i in range(1, len(CLASSES)):
            nc_organ_dir = os.path.join(nc_root, CLASSES[i])
            pv_organ_dir = os.path.join(pv_root, CLASSES[i])

            nc_train_root = os.path.join(nc_organ_dir, 'train')
            nc_valid_root = os.path.join(nc_organ_dir, 'valid')

            pv_train_root = os.path.join(pv_organ_dir, 'train')
            pv_valid_root = os.path.join(pv_organ_dir, 'valid')

            ## Input training data
            # nc_train_label_dir = os.path.join(nc_train_root, 'label_patches_%d'%(i))
            nc_train_prior_dir = os.path.join(nc_train_root, 'prior_patches_%d'%(i))
            nc_train_img_dir = os.path.join(nc_train_root, 'softimg_patches_%d'%(i))
            # nc_label_train = nc_label_train + glob.glob(nc_train_label_dir, '*.png')
            nc_prior_train = nc_prior_train + glob.glob(os.path.join(nc_train_prior_dir, '*.png'))
            nc_img_train = nc_img_train + glob.glob(os.path.join(nc_train_img_dir, '*.png'))

            # pv_train_label_dir = os.path.join(pv_train_root, 'label_patches_%d'%(i))
            pv_train_prior_dir = os.path.join(pv_train_root, 'prior_patches_%d'%(i))
            pv_train_img_dir = os.path.join(pv_train_root, 'softimg_patches_%d'%(i))
            # pv_label_train = nc_label_train + glob.glob(pv_train_label_dir, '*.png')
            pv_prior_train = pv_prior_train + glob.glob(os.path.join(pv_train_prior_dir, '*.png'))
            pv_img_train = pv_img_train + glob.glob(os.path.join(pv_train_img_dir, '*.png'))

            ## Input validation data
            # nc_valid_label_dir = os.path.join(nc_valid_root, 'label_patches_%d'%(i))
            # nc_valid_prior_dir = os.path.join(nc_valid_root, 'prior_patches_%d'%(i))
            # nc_valid_img_dir = os.path.join(nc_valid_root, 'softimg_patches_%d'%(i))
            # nc_label_valid = nc_label_valid + glob.glob(nc_valid_label_dir, '*.png')
            # nc_prior_valid = nc_prior_valid + glob.glob(nc_valid_prior_dir, '*.png')
            # nc_img_valid = nc_img_valid + glob.glob(nc_valid_img_dir, '*.png')

            # pv_valid_label_dir = os.path.join(pv_valid_root, 'label_patches_%d'%(i))
            # pv_valid_prior_dir = os.path.join(pv_valid_root, 'prior_patches_%d'%(i))
            # pv_valid_img_dir = os.path.join(pv_valid_root, 'softimg_patches_%d'%(i))
            # pv_label_valid = nc_label_valid + glob.glob(pv_valid_label_dir, '*.png')
            # pv_prior_valid = nc_prior_valid + glob.glob(pv_valid_prior_dir, '*.png')
            # pv_img_valid = nc_img_valid + glob.glob(pv_valid_img_dir, '*.png')
            print('Finished Loading Organ Dataset: {}'.format(CLASSES[i]))
        # art_root = os.path.join('/nfs/masi/leeh43/contrastive_learning/SupContrast/rp_training/spleen/art')
        train_dataset = Data_loader(pv_prior_train, pv_img_train, nc_prior_train, nc_img_train, train=True)
        # valid_dataset = Data_loader(pv_valid_root, nc_valid_root, train=False)
        print('Finished Loading All Organ Dataset')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    # valid_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=opt.batch_size, shuffle=False,
    #     num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    if opt.method == 'CrossEntropy':
        model =  SupCEResNet(name=opt.model)
    elif opt.method == 'SupCon' or opt.method == 'SimCLR':
        model =  SupConResNet(name=opt.model)
    elif opt.method == 'DenseCon':
        model = DenseResNet(name=opt.model)
    criterion = DCCLoss(temperature=opt.temp)
    # criterion = torch.nn.BCELoss()
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    log_file = os.path.join(opt.model_path, 'training_loss.txt')
    fv = open(log_file, 'a')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    

    end = time.time()
    # print(train_loader)
    for idx, (images, priors, labels, img_name) in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc='Train epoch=%d' % epoch, ncols=80, leave=False):
        data_time.update(time.time() - end)
                
        # print(img_name)
        # print(images[0].shape)
        img_1 = torch.cat([images[0], priors[0]], dim=1)
        img_2 = torch.cat([images[1], priors[1]], dim=1)
        images_f = torch.cat([img_1, img_2], dim=0).cuda(non_blocking=True)
        images_f = images_f.cuda(non_blocking=True)
        # images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = int(images_f.shape[0] / 2)
        
        mask_1 = images[0] * priors[0]
        mask_2 = images[1] * priors[1]
        
        image_weight = torch.zeros((images_f.size(0),images_f.size(0)))
        for i in range(img_1.size(0) * 2):
            if i < img_1.size(0):
                tmp_1 = mask_1[i, :, :, :].cpu().data.numpy()
                if np.sum(tmp_1) != 0:
                    pixel_count = np.count_nonzero(tmp_1)
                    mean_p_1 = np.sum(tmp_1) / pixel_count
                else:
                    mean_p_1 = 0
            else:
                tmp_1 = mask_2[i-img_1.size(0), :, :, :].cpu().data.numpy()
                if np.sum(tmp_1) != 0:
                    pixel_count = np.count_nonzero(tmp_1)
                    mean_p_1 = np.sum(tmp_1) / pixel_count
                else:
                    mean_p_1 = 0

            for j in range(img_1.size(0)):
                tmp_2 = mask_1[j, :, :, :].cpu().data.numpy()
                if np.sum(tmp_2) == 0:
                    image_weight[i, j] = torch.tensor(np.array(0)).unsqueeze(0)
                else:
                    pixel_count = np.count_nonzero(tmp_2)
                    mean_p_2 = np.sum(tmp_2) / pixel_count
                    
                    inten_diff = abs(mean_p_1 - mean_p_2)
                    # print(inten_diff)
                    image_weight[i, j] = torch.tensor(np.array(inten_diff)).unsqueeze(0)
                
                tmp_3 = mask_2[j, :, :, :].cpu().data.numpy()
                if np.sum(tmp_3) == 0:
                    image_weight[i, j] = torch.tensor(np.array(0)).unsqueeze(0)
                else:
                    pixel_count = np.count_nonzero(tmp_3)
                    mean_p_3 = np.sum(tmp_3) / pixel_count
                    inten_diff_2 = abs(mean_p_1 - mean_p_3)
                    # print(inten_diff_2)
                    
                    image_weight[i, j+img_1.size(0)] = torch.tensor(np.array(inten_diff_2)).unsqueeze(0)


        image_weight = 1 - image_weight
        # norm_img_w += eps
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images_f)
        f1, f2 = torch.split(features[0], [img_1.size(0), img_1.size(0)], dim=0)
        g_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        # dense_f1, dense_f2 = torch.split(features[1], [bsz, bsz], dim=0)
        # dense_feat = torch.cat([dense_f1, dense_f2], dim=1)
        # print(features.shape)
        # features = F.softmax(features)
        if opt.method == 'SupCon':
            loss = criterion(g_feat, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(g_feat, image_weight=image_weight.cuda())
        elif opt.method == 'CrossEntropy':
            loss_1 = criterion(f1.unsqueeze(1), labels)
            loss_2 = criterion(f2.unsqueeze(1), labels)
            loss = (loss_1 + loss_2) / 2
        # elif opt.method == 'DenseCon':
        #     ## SupCon
        #     loss_glob = criterion(g_feat, labels)
        #     loss_dense = criterion(dense_feat, labels)
            
        #     loss = loss_dense*0.2 + loss_glob*0.8 + 1e-6
        #     ## Dense Loss
        #     # res_f_1, res_f_2 = 
            
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
        
        fv.write('epoch=%d,  batch_idx=%d, loss=%.4f \n' % (epoch, idx, loss.data))

    return losses.avg


def valid(valid_loader, model, criterion, optimizer, epoch, opt):
    model.eval()

    log_file = os.path.join(opt.model_path, 'valid_loss.txt')
    fv = open(log_file, 'a')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, img_name) in tqdm.tqdm(
                enumerate(valid_loader), total=len(valid_loader),
                desc='valid epoch=%d' % epoch, ncols=80, leave=False):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = int(images.shape[0] / 2)

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Valid: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(valid_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
        
        fv.write('epoch=%d,  batch_idx=%d, loss=%.4f \n' % (epoch, idx, loss.data))


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    print(opt.epochs)
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # time3 = time.time()
        # valid_loss = valid(valid_loader, model, criterion, optimizer, epoch, opt)
        # time4 = time.time()
        # print('epoch {}, total time {:.2f}'.format(epoch, time4 - time3))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
