from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utils import AverageMeter
from utils import adjust_learning_rate, warmup_learning_rate, accuracy
from utils import set_optimizer
from networks.resnet_big_6channels import SupCEResNet, SupConResNet, LinearClassifier, SegmentationModel, DenseResNet

import os
from Data_loader_segmentation import Data_loader
import torch.nn.functional as F
import tqdm
import numpy as np
from PIL import Image
import glob
# import monai
os.environ["CUDA_VISIBLE_DEVICES"]='0'


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


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
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='imagevu',
                        choices=['cifar10', 'cifar100', 'imagevu'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='/nfs/masi/leeh43/Supcon_learning/code/save/SupCon/imagevu_temp_0.07_models_128x128_pv_nc_resnet50_lr_0.0005_bz_4_6channel_simclr_3/SimCLR_imagevu_resnet50_lr_0.0005_decay_0.0001_bsz_4_temp_0.07_trial_0/ckpt_epoch_2.pth',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagevu':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def dice_loss(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    probs = F.softmax(input)
    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=0)   # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=0)  # b,c,1,1

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=0)  # b,c,1,1

    dice = 2 * ((num+0.0000001) / (den1 + den2+0.0000001))
    dice_eso = dice[1:]  # we ignore bg dice val, and take the fg

    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return dice_total


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    classifier = SegmentationModel(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        # if torch.cuda.device_count() > 1:
        #     model.encoder = torch.nn.DataParallel(model.encoder)
        # else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, criterion, classifier

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
            nc_train_label_dir = os.path.join(nc_train_root, 'label_patches_%d'%(i))
            nc_train_prior_dir = os.path.join(nc_train_root, 'prior_patches_%d'%(i))
            nc_train_img_dir = os.path.join(nc_train_root, 'softimg_patches_%d'%(i))
            nc_label_train = nc_label_train + glob.glob(os.path.join(nc_train_label_dir, '*.png'))
            nc_prior_train = nc_prior_train + glob.glob(os.path.join(nc_train_prior_dir, '*.png'))
            nc_img_train = nc_img_train + glob.glob(os.path.join(nc_train_img_dir, '*.png'))

            pv_train_label_dir = os.path.join(pv_train_root, 'label_patches_%d'%(i))
            pv_train_prior_dir = os.path.join(pv_train_root, 'prior_patches_%d'%(i))
            pv_train_img_dir = os.path.join(pv_train_root, 'softimg_patches_%d'%(i))
            pv_label_train = pv_label_train + glob.glob(os.path.join(pv_train_label_dir, '*.png'))
            pv_prior_train = pv_prior_train + glob.glob(os.path.join(pv_train_prior_dir, '*.png'))
            pv_img_train = pv_img_train + glob.glob(os.path.join(pv_train_img_dir, '*.png'))

            ## Input validation data
            nc_valid_label_dir = os.path.join(nc_valid_root, 'label_patches_%d'%(i))
            nc_valid_prior_dir = os.path.join(nc_valid_root, 'prior_patches_%d'%(i))
            nc_valid_img_dir = os.path.join(nc_valid_root, 'softimg_patches_%d'%(i))
            nc_label_valid = nc_label_valid + glob.glob(os.path.join(nc_valid_label_dir, '*.png'))
            nc_prior_valid = nc_prior_valid + glob.glob(os.path.join(nc_valid_prior_dir, '*.png'))
            nc_img_valid = nc_img_valid + glob.glob(os.path.join(nc_valid_img_dir, '*.png'))

            pv_valid_label_dir = os.path.join(pv_valid_root, 'label_patches_%d'%(i))
            pv_valid_prior_dir = os.path.join(pv_valid_root, 'prior_patches_%d'%(i))
            pv_valid_img_dir = os.path.join(pv_valid_root, 'softimg_patches_%d'%(i))
            pv_label_valid = pv_label_valid + glob.glob(os.path.join(pv_valid_label_dir, '*.png'))
            pv_prior_valid = pv_prior_valid + glob.glob(os.path.join(pv_valid_prior_dir, '*.png'))
            pv_img_valid = pv_img_valid + glob.glob(os.path.join(pv_valid_img_dir, '*.png'))
            print('Finished Loading Organ Dataset: {}'.format(CLASSES[i]))
        # art_root = os.path.join('/nfs/masi/leeh43/contrastive_learning/SupContrast/rp_training/spleen/art')
        train_dataset = Data_loader(pv_label_train, pv_prior_train, pv_img_train, nc_label_train, nc_prior_train, nc_img_train, train=True)
        valid_dataset = Data_loader(pv_label_valid, pv_prior_valid, pv_img_valid, nc_label_valid, nc_prior_valid, nc_img_valid, train=False)
        print('Finished Loading All Organ Dataset')

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, valid_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()
    
    model_dir = os.path.join('/nfs/masi/leeh43/Supcon_learning/rp_resnet50_all_organs_supcon_6channel_simclr_3')
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    model_pth = '%s/model_epoch_%04d.pth' % (model_dir, epoch)
    log_file = os.path.join(model_dir, 'training_loss_segmentation.txt')
    fv = open(log_file, 'a')

    # loss_function = monai.losses.DiceLoss(include_background=False, softmax=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels, l_vol, name) in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                desc='Train epoch=%d' % epoch, ncols=80, leave=False):
        data_time.update(time.time() - end)

        images = Variable(images.cuda(non_blocking=True))
        labels = Variable(labels.cuda(non_blocking=True))
        l_vol = Variable(l_vol.cuda(non_blocking=True))
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        # loss = criterion(output, labels)
        # print(output.shape)
        # print(l_vol.shape)
        loss = dice_loss(output, l_vol)
        # print('Dice loss = {}'.format(dice))
        # loss += dice
        print('epoch=%d,  batch_idx=%d, loss=%.4f \n' % (epoch, idx, loss.data))
        
        fv.write('epoch=%d,  batch_idx=%d, loss=%.4f \n' % (epoch, idx, loss.data))


        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return losses


def validate(val_loader, model, criterion, classifier, epoch, opt):
    """validation"""
    
    colorpick = [[0,0,0], [255,30,30],[255,245,71],[112,255,99],[9,150,37],[30,178,252],[132,0,188],\
        [255,81,255],[158,191,9],[255,154,2],[102,255,165],[0,242,209],[255,0,80],[255,0,160],[100,100,100],[170,170,170],[230,230,230]]
    colorpick = np.array(colorpick)
    cmap = colorpick.flatten().tolist()
        
    model.eval()
    classifier.eval()
    
    model_dir = os.path.join('/nfs/masi/leeh43/Supcon_learning/rp_resnet50_all_organs_supcon_6channel_simclr_3')
    log_file = os.path.join(model_dir, 'validation_loss_segmentation.txt')
    fv = open(log_file, 'a')
    # loss_function = monai.losses.DiceLoss(softmax=True)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, l_vol, name) in tqdm.tqdm(
                enumerate(val_loader), total=len(val_loader),
                desc='Valid epoch=%d' % epoch, ncols=80, leave=False):
            images = images.cuda()
            labels = labels.cuda()
            l_vol = l_vol.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            # print(output.size())
            # loss = criterion(output, labels)
            loss = dice_loss(output, l_vol)
            # print('Dice loss = {}'.format(dice))
            # loss += dice
            _, pred = torch.max(output, 1)
            pred = pred.data.cpu().numpy().astype(np.uint8)
            print(pred.shape)
            print(np.unique(pred))
            
            for i in range(pred.shape[0]):
                imname = name[i]
                mask_pred = Image.fromarray(pred[i,:,:])
                mask_pred.putpalette(cmap)
                # mask_pred = mask_pred.resize((int(512), int(512)))
                outlabel_dir = os.path.join(model_dir, 'val_result')
                if not os.path.isdir(outlabel_dir):
                    os.makedirs(outlabel_dir)
                mask_pred.save(os.path.join(outlabel_dir, imname))
            print('epoch=%d,  batch_idx=%d, loss=%.4f \n' % (epoch, idx, loss.data))
            
            fv.write('epoch=%d,  batch_idx=%d, loss=%.4f \n' % (epoch, idx, loss.data))

    return losses


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion, classifier = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    model_dir = os.path.join('/nfs/masi/leeh43/Supcon_learning/rp_resnet50_all_organs_supcon_6channel_simclr_3')
    # model_pth = os.path.join('/nfs/masi/leeh43/Supcon_learning/rp_resnet50_all_organs_DenseCon_0.8_0.2_3/model_epoch_0008.pth')
    # classifier.load_state_dict(torch.load(model_pth))
    # training routine
    for epoch in range(0, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        model_pth = '%s/model_epoch_%04d.pth' % (model_dir, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}'.format(
            epoch, time2 - time1))

        # eval for one epoch
        loss = validate(val_loader, model, criterion, classifier, epoch, opt)
        
        # Save model
        enc_pth = os.path.join(model_dir, 'enc_model_epoch_%04d.pth' % (epoch))
        torch.save(classifier.state_dict(), model_pth)
        # torch.save(model.state_dict(), enc_pth)



if __name__ == '__main__':
    main()
