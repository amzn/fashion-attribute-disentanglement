# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os
import numpy as np
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataloader import Data, DataQuery
from model import Extractor, MemoryBlock
from argument_parser import add_base_args, add_eval_args
from utils import split_labels,  compute_NDCG, get_target_attr
import constants as C

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_eval_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        torch.cuda.set_device(args.gpu_id)

    file_root = args.file_root
    img_root_path = args.img_root

    # load dataset
    print('Loading gallery...')
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

    model = Extractor(gallery_data.attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)
    memory = MemoryBlock(gallery_data.attr_num)

    if args.load_pretrained_extractor:
        print('load {path} \n'.format(path=args.load_pretrained_extractor))
        model.load_state_dict(torch.load(args.load_pretrained_extractor))
    else:
        print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')
    if args.load_pretrained_memory:
        print('load {path} \n'.format(path=args.load_pretrained_memory))
        memory.load_state_dict(torch.load(args.load_pretrained_memory))
    else:
        print('Pretrained memory not provided. Use --load_pretrained_memory or the model will be randomly initialized.')
    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir)

    if not args.use_cpu:
        model.cuda()
        memory.cuda()

    model.eval()
    memory.eval()

    #indexing the gallery
    gallery_feat = []
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(gallery_loader)):
            if not args.use_cpu:
                img = img.cuda()

            dis_feat, _ = model(img)
            gallery_feat.append(F.normalize(torch.cat(dis_feat, 1)).squeeze().cpu().numpy())

    if args.save_matrix:
        np.save(os.path.join(args.feat_dir, 'gallery_feats.npy'), np.concatenate(gallery_feat, axis=0))
        print('Saved indexed features at {dir}/gallery_feats.npy'.format(dir=args.feat_dir))
    #indexing the query
    query_inds = np.loadtxt(os.path.join(file_root, args.query_inds), dtype=int)
    gt_labels = np.loadtxt(os.path.join(file_root, args.gt_labels), dtype=int)
    ref_idxs = np.loadtxt(os.path.join(file_root, args.ref_ids), dtype=int)

    assert (query_inds.shape[0] == gt_labels.shape[0]) and (query_inds.shape[0] == ref_idxs.shape[0])

    query_fused_feats = []
    print('Loading test queries...')
    query_data = DataQuery(file_root, img_root_path,
                           args.ref_ids, args.query_inds,
                           transforms.Compose([
                               transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                           ]), mode='test')
    query_loader = torch.utils.data.DataLoader(query_data, batch_size=args.batch_size, shuffle=False,
                                   sampler=torch.utils.data.SequentialSampler(query_data),
                                   num_workers=args.num_threads,
                                   drop_last=False)

    with torch.no_grad():
        for i, (img, indicator) in enumerate(tqdm(query_loader)):
            indicator = indicator.float()
            if not args.use_cpu:
                img = img.cuda()
                indicator = indicator.cuda()

            dis_feat, _ = model(img)
            residual_feat = memory(indicator)
            feat_manip = torch.cat(dis_feat, 1) + residual_feat

            query_fused_feats.append(F.normalize(feat_manip).cpu().numpy())

    if args.save_matrix:
        np.save(os.path.join(args.feat_dir, 'query_fused_feats.npy'), np.concatenate(query_fused_feats, axis=0))
        print('Saved query features at {dir}/query_fused_feats.npy'.format(dir=args.feat_dir))

    #evaluate the top@k results
    gallery_feat = np.concatenate(gallery_feat, axis=0).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    fused_feat = np.array(np.concatenate(query_fused_feats, axis=0)).reshape(-1, args.dim_chunk * len(gallery_data.attr_num))
    dim = args.dim_chunk * len(gallery_data.attr_num)  # dimension
    num_database = gallery_feat.shape[0]  # number of images in database
    num_query = fused_feat.shape[0]  # number of queries

    database = gallery_feat
    queries = fused_feat
    index = faiss.IndexFlatL2(dim)
    index.add(database)
    k = args.top_k
    _, knn = index.search(queries, k)

    #load the GT labels for all gallery images
    label_data = np.loadtxt(os.path.join(file_root, 'labels_test.txt'), dtype=int)

    #compute top@k acc
    hits = 0
    for q in tqdm(range(num_query)):
        neighbours_idxs = knn[q]
        for n_idx in neighbours_idxs:
            if (label_data[n_idx] == gt_labels[q]).all():
                hits += 1
                break
    print('Top@{k} accuracy: {acc}'.format(k=k, acc=hits/num_query))

    #compute NDCG
    ndcg = []
    ndcg_target = []  # consider changed attribute only
    ndcg_others = []  # consider other attributes

    for q in tqdm(range(num_query)):
        rel_scores = []
        target_scores = []
        others_scores = []

        neighbours_idxs = knn[q]
        indicator = query_inds[q]
        target_attr = get_target_attr(indicator, gallery_data.attr_num)
        target_label = split_labels(gt_labels[q], gallery_data.attr_num)

        for n_idx in neighbours_idxs:
            n_label = split_labels(label_data[n_idx], gallery_data.attr_num)
            # compute matched_labels number
            match_cnt = 0
            others_cnt = 0

            for i in range(len(n_label)):
                if (n_label[i] == target_label[i]).all():
                    match_cnt += 1
                if i == target_attr:
                    if (n_label[i] == target_label[i]).all():
                        target_scores.append(1)
                    else:
                        target_scores.append(0)
                else:
                    if (n_label[i] == target_label[i]).all():
                        others_cnt += 1

            rel_scores.append(match_cnt / len(gallery_data.attr_num))
            others_scores.append(others_cnt / (len(gallery_data.attr_num) - 1))

        ndcg.append(compute_NDCG(np.array(rel_scores)))
        ndcg_target.append(compute_NDCG(np.array(target_scores)))
        ndcg_others.append(compute_NDCG(np.array(others_scores)))

    print('NDCG@{k}: {ndcg}, NDCG_target@{k}: {ndcg_t}, NDCG_others@{k}: {ndcg_o}'.format(k=k,
                                                                                          ndcg=np.mean(ndcg),
                                                                                          ndcg_t=np.mean(ndcg_target),
                                                                                          ndcg_o=np.mean(ndcg_others)))
