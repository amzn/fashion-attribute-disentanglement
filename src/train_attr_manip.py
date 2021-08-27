# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
This is script is used to joint train the disentangled representation learner and memory block.
"""

import argparse
import datetime
import json

import constants as C
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from argument_parser import add_base_args, add_train_args
from dataloader import DataTriplet, DataQuery, Data
from loss_function import hash_labels, TripletSemiHardLoss
from model import Extractor, MemoryBlock
from tqdm import tqdm
from utils import get_non_diagonal_elements
import faiss

torch.manual_seed(100)


def train(train_loader, model, memory, optimizer, args):
    avg_total_loss = 0

    model.train()
    memory.train()

    for i, (imgs, one_hots, labels, indicator) in enumerate(tqdm(train_loader)):
        indicator = indicator.float()
        for key in one_hots.keys():
            one_hots[key] = one_hots[key].float()
        if not args.use_cpu:
            for key in imgs.keys():
                imgs[key] = imgs[key].cuda()
                one_hots[key] = one_hots[key].cuda()
                labels[key] = labels[key].cuda()
            indicator = indicator.cuda()

        model.zero_grad()
        memory.zero_grad()

        feats = {}
        cls_outs = {}
        for key in imgs.keys():
            feats[key], cls_outs[key] = model(imgs[key])

        residual_feat = memory(indicator)
        feat_manip = torch.cat(feats['ref'], 1) + residual_feat
        feat_manip_split = list(torch.split(feat_manip, args.dim_chunk, dim=1))

        cls_outs_manip = []
        for attr_id, layer in enumerate(model.attr_classifier):
            cls_outs_manip.append(layer(feat_manip_split[attr_id]).squeeze())

        # attribute prediction loss
        cls_loss = 0
        for j in range(len(train_loader.dataset.attr_num)):
            for key in imgs.keys():
                cls_loss += F.cross_entropy(cls_outs[key][j], labels[key][:, j], ignore_index=-1)
            cls_loss += F.cross_entropy(cls_outs_manip[j], labels['pos'][:, j], ignore_index=-1)

        # label_triplet_loss
        hashs = {}
        for key in imgs.keys():
            hashs[key] = hash_labels(labels[key])

        label_triplet_loss = TripletSemiHardLoss(torch.cat((hashs['ref'], hashs['pos'], hashs['neg']), 0),
                                                  torch.cat((F.normalize(torch.cat(feats['ref'], 1)),
                                                             F.normalize(feat_manip),
                                                             F.normalize(torch.cat(feats['neg'], 1))), 0),
                                                  margin=args.margin)

        # manipulation_triplet_loss
        criterion_c = nn.TripletMarginLoss(margin=args.margin)
        manip_triplet_loss = criterion_c(F.normalize(feat_manip),
                                         F.normalize(torch.cat(feats['pos'], 1)),
                                         F.normalize(torch.cat(feats['neg'], 1))
                                         )
        total_loss = args.weight_cls * cls_loss + args.weight_label_trip * label_triplet_loss + args.weight_manip_trip * manip_triplet_loss

        # consistent loss
        if args.consist_loss:
            consist_loss = 0
            propotypes = {}
            for key in imgs.keys():
                propotypes[key] = memory(one_hots[key])

            consist_loss += F.l1_loss(propotypes['pos'], feat_manip)
            consist_loss += F.l1_loss(propotypes['ref'], torch.cat(feats['ref'], 1))
            consist_loss += F.l1_loss(propotypes['neg'], torch.cat(feats['neg'], 1))
            total_loss += args.weight_consist * consist_loss

        # diagonal regularization
        if args.diagonal:
            N = get_non_diagonal_elements(train_loader.dataset.attr_num, args.dim_chunk)
            if not args.use_cpu:
                N = N.cuda()
            total_loss += args.weight_diag * (torch.norm(N * memory.Memory.weight, 1))

        total_loss.backward()
        optimizer.step()

        avg_total_loss += total_loss.data

    return avg_total_loss / (i+1)


def eval(gallery_loader, query_loader, model, memory, args):
    model.eval()
    memory.eval()

    gt_labels = np.loadtxt(os.path.join(args.file_root, 'gt_test.txt'), dtype=int)
   
    gallery_feat = []
    query_fused_feats = []
    with torch.no_grad():
         # indexing the gallery
        for i, (img, _) in enumerate(tqdm(gallery_loader)):
            if not args.use_cpu:
                img = img.cuda()

            dis_feat, _ = model(img)
            gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

        # load the queries
        for i, (img, indicator) in enumerate(tqdm(query_loader)):
            indicator = indicator.float()
            if not args.use_cpu:
                img = img.cuda()
                indicator = indicator.cuda()

            dis_feat, _ = model(img)
            residual_feat = memory(indicator)
            feat_manip = torch.cat(dis_feat, 1) + residual_feat

            query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())


    gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, args.dim_chunk * len(gallery_loader.dataset.attr_num))
    fused_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(gallery_loader.dataset.attr_num))
    dim = args.dim_chunk * len(gallery_loader.dataset.attr_num)  # dimension
    num_query = fused_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = fused_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = 30
    _, knn = index.search(queries, k)

    # load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)

    # compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits/num_query))

    return hits/num_query


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
    train_data = DataTriplet(file_root, img_root_path, args.triplet_file,
                             transforms.Compose([
                                 transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), 'train')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_threads,
                                               drop_last=True)

    query_data = DataQuery(file_root, img_root_path,
                           'ref_test.txt', 'indfull_test.txt',
                           transforms.Compose([
                               transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                           ]), mode='test')
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size, shuffle=False,
                                   sampler=torch.utils.data.SequentialSampler(query_data),
                                   num_workers=args.num_threads,
                                   drop_last=False)

    gallery_data = Data(file_root, img_root_path,
                        transforms.Compose([
                            transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                        ]), mode='test')

    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=args.batch_size, shuffle=False,
                                     sampler=torch.utils.data.SequentialSampler(gallery_data),
                                     num_workers=args.num_threads,
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
    memory = MemoryBlock(train_data.attr_num)

    # start training from the pretrained weights if provided
    if args.load_pretrained_extractor:
        print('load %s\n' % args.load_pretrained_extractor)
        model.load_state_dict(torch.load(args.load_pretrained_extractor))
    else:
        print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')
    if args.load_pretrained_memory:
        print('load %s\n' % args.load_pretrained_memory)
        memory.load_state_dict(torch.load(args.load_pretrained_memory))
    else:
        print('Pretrained memory not provided. Use --load_pretrained_memory or the model will be randomly initialized.')

    if args.load_init_mem:
        print('load initial memory block %s\n' % args.load_init_mem)
        init_mem_np = np.load(args.load_init_mem)
        with torch.no_grad():
            assert (memory.Memory.weight.shape == init_mem_np.shape)
            memory.Memory.weight = nn.Parameter(torch.from_numpy(init_mem_np).float())

    if not args.use_cpu:
        model.cuda()
        memory.cuda()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(memory.parameters()), lr=args.lr,
                                 betas=(args.momentum, 0.999))
    lr_scheduler = lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_rate)

    previous_best_avg_test_acc = 0.0
    for epoch in range(args.num_epochs):
        avg_total_loss = train(train_loader, model, memory, optimizer, args)
        avg_test_acc = eval(gallery_loader, query_loader, model, memory, args)

        # result record
        print('Epoch %d, Cls_loss: %.4f, test_acc: %.4f\n' % (epoch + 1, avg_total_loss, avg_test_acc))

        with open(os.path.join(directory, 'log.txt'), 'a') as f:
            f.write('Epoch %d, Cls_loss: %.4f, test_acc: %.4f\n' % (epoch + 1, avg_total_loss, avg_test_acc))

        # store parameters
        torch.save(model.state_dict(), os.path.join(directory, "extractor_ckpt_%d.pkl" % (epoch + 1)))
        torch.save(memory.state_dict(), os.path.join(directory, "memory_ckpt_%d.pkl" % (epoch + 1)))
        print('Saved checkpoints at {dir}/extractor_{epoch}.pkl, {dir}/memory_{epoch}.pkl'.format(dir=directory,
                                                                                                  epoch=epoch + 1))
        if avg_test_acc > previous_best_avg_test_acc:
            torch.save(model.state_dict(), os.path.join(directory, "extractor_best.pkl"))
            torch.save(memory.state_dict(), os.path.join(directory, "memory_best.pkl"))
            print('Best model in {dir}/extractor_best.pkl and {dir}/memory_best.pkl'.format(dir=directory))
            previous_best_avg_test_acc = avg_test_acc

        lr_scheduler.step()
