# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
This is script is used to initialize the memory block. We load the pretrained weights of
disentangled representation learner, and then average the attribute embedding of the training images
with the same attribute values as the initial prototype embedding and store them in the memory block.
"""

import argparse
import os
import numpy as np
import random
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from model import Extractor
from argument_parser import add_base_args, add_init_args
import constants as C

torch.manual_seed(100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    add_init_args(parser)
    args = parser.parse_args()
    if not args.use_cpu and not torch.cuda.is_available():
        print('Warning: Using CPU')
        args.use_cpu = True
    else:
        torch.cuda.set_device(args.gpu_id)

    file_root = args.file_root
    img_root_path = args.img_root

    attr_num = np.loadtxt(os.path.join(file_root, "attr_num.txt"), dtype=int)

    model = Extractor(attr_num, backbone=args.backbone, dim_chunk=args.dim_chunk)

    if args.load_pretrained_extractor:
        print('load {path}\n'.format(path=args.load_pretrained_extractor))
        model.load_state_dict(torch.load(args.load_pretrained_extractor))
    else:
        print('Pretrained extractor not provided. Use --load_pretrained_extractor or the model will be randomly initialized.')
    if not args.use_cpu:
        model.cuda()

    # load training data
    with open(os.path.join(file_root, 'imgs_train.txt')) as f:
        imgs_train = f.read().splitlines()
    labels_train = np.loadtxt(os.path.join(file_root, "labels_train.txt"), dtype=int)
    assert len(imgs_train) == labels_train.shape[0]

    model.eval()

    #build a check-up table so given the idx of attribute values we know which attribute it's belonged to
    idx2type = []
    for i, attr_cnt in enumerate(attr_num):
        idx2type += [i] * attr_cnt

    attr_sum = sum(attr_num)  # number of all attribute values
    vector_dim = len(attr_num) * args.dim_chunk  # the dimension of each prototype embedding

    memory = np.zeros((vector_dim, attr_sum))

    with torch.no_grad():
        for i in tqdm(range(attr_sum)):
            attr_candis = np.where(labels_train[:, i])[0]
            type_attr = idx2type[i]
            feat = []
            for k in range(args.num_sample):
                idx = random.sample(list(attr_candis), 1)[0]
                path = imgs_train[idx]
                ref_img = Image.open(os.path.join(img_root_path, path)).convert('RGB')

                img_transform = transforms.Compose([
                    transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                ])

                ref_img = img_transform(ref_img).unsqueeze(0)
                if not args.use_cpu:
                    ref_img = ref_img.cuda()
                feat_ref, _ = model(ref_img)
                feat.append(feat_ref[type_attr].cpu().numpy())

            memory[type_attr*args.dim_chunk:type_attr*args.dim_chunk+args.dim_chunk, i] = np.mean(np.array(feat), 0)

    if not os.path.exists(args.memory_dir):
        os.makedirs(args.memory_dir)
    np.save(os.path.join(args.memory_dir, 'init_mem.npy'), np.array(memory))
    print('initialized memory block saved at {output_dir}/init_mem.npy'.format(output_dir=args.memory_dir))

