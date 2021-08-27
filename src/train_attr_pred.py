# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
This is script is used to pre-train the disentangled representation learner with attribute clossification task
"""

import argparse
import datetime
import json

import constants as C
import os
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from argument_parser import add_base_args, add_train_args
from dataloader import Data
from loss_function import hash_labels, TripletSemiHardLoss
from model import Extractor
from tqdm import tqdm

torch.manual_seed(100)


def train(train_loader, model, optimizer, args):
    avg_total_loss = 0
    model.train()
    for i, sample in enumerate(tqdm(train_loader)):
        img_query, label = sample
        if not args.use_cpu:
            img_query = img_query.cuda()
            label = label.cuda()

        model.zero_grad()
        dis_feat, cls_outs = model(img_query)

        cls_loss = 0
        for j in range(len(train_loader.dataset.attr_num)):
            cls_loss += F.cross_entropy(cls_outs[j], label[:, j], ignore_index=-1)

        # attr_rank_loss_local
        rank_loss = 0
        for j in range(len(train_loader.dataset.attr_num)):
            rank_loss += TripletSemiHardLoss(label[:, j], F.normalize(dis_feat[j]), margin=args.margin)

        # attr_rank_loss_global
        hash_label = hash_labels(label)
        rank_global_loss = TripletSemiHardLoss(hash_label, F.normalize(torch.cat(dis_feat, 1)),
                                               margin=args.margin)

        total_loss = args.weight_cls * cls_loss + args.weight_label_trip_local * rank_loss + args.weight_label_trip * rank_global_loss
        total_loss.backward()
        optimizer.step()

        avg_total_loss += total_loss

    return avg_total_loss / (i+1)


def eval(test_loader, model, args):
    model.eval()
    attr_num = test_loader.dataset.attr_num
    total = 0
    hit = 0

    for i, sample in enumerate(tqdm(test_loader)):
        img_query, label = sample
        if not args.use_cpu:
            img_query = img_query.cuda()
            label = label.cuda()
        _, cls_outs = model(img_query)


        for j in range(len(attr_num)):
            for b in range(img_query.shape[0]):
                gt = label[b, j]
                if gt != -1:
                    total += 1
                    pred = torch.argmax(cls_outs[j][b])
                    if pred == gt:
                        hit += 1

    return hit/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        torch.cuda.set_device(args.gpu_id)

    file_root = args.file_root
    img_root_path = args.img_root

    # load dataset
    print('Loading dataset...')
    train_data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                          transforms.RandomHorizontalFlip(),
                          transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'train')
    valid_data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'test')

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_threads,
                                   drop_last=True)
    valid_loader = data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_threads,
                                   drop_last=False)

    # create the folder to save log, checkpoints and args config
    if not args.ckpt_dir:
        name = datetime.datetime.now().strftime("%m-%d-%H:%M")
    else:
        name = args.ckpt_dir
    directory = '{name}'.format(name=name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    model = Extractor(train_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)

    if args.load_pretrained_extractor:
        print('load %s\n' % args.load_pretrained_extractor)
        model.load_state_dict(torch.load(args.load_pretrained_extractor))
    else:
        print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')

    if not args.use_cpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    lr_scheduler = lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_rate)

    previous_best_avg_test_acc = 0.0
    for epoch in range(args.num_epochs):
        avg_total_loss = train(train_loader, model, optimizer, args)
        avg_test_acc = eval(valid_loader, model, args)

        # result record
        print('Epoch %d, Train_loss: %.4f,  test_acc: %.4f \n'
              % (epoch + 1, avg_total_loss, avg_test_acc))

        with open(os.path.join(directory, 'log.txt'), 'a') as f:
            f.write('Epoch %d, Train_loss: %.4f, test_acc: %.4f\n'
                    % (epoch + 1, avg_total_loss, avg_test_acc))

        # store parameters
        torch.save(model.state_dict(), os.path.join(directory, "ckpt_%d.pkl" % (epoch + 1)))
        print('Saved checkpoints at {dir}/ckpt_{epoch}.pkl'.format(dir=directory, epoch=epoch+1))

        if avg_test_acc > previous_best_avg_test_acc:
            torch.save(model.state_dict(), os.path.join(directory, "extractor_best.pkl"))
            print('Best model in {dir}/extractor_best.pkl'.format(dir=directory))
            previous_best_avg_test_acc = avg_test_acc

        lr_scheduler.step()
