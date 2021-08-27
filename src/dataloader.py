# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import numpy as np
from PIL import Image, ImageFile
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import get_idx_label


class DataQuery(data.Dataset):
    """
    Load generated queries for evaluation. Each query consists of a reference image and an indicator vector
    The indicator vector consists of -1, 1 and 0, which means remove, add, not modify
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_test.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        ref_ids: the file name of the generated txt file, which includes the indices of reference images
        query_inds: the file name of the generated txt file, which includes the indicator vector for queries.
        img_transform: transformation functions for img. Default: ToTensor()
        mode: the mode 'train' or 'test' decides to load training set or test set
    """
    def __init__(self, file_root,  img_root_path, ref_ids,  query_inds, img_transform=None,
                 mode='test'):
        super(DataQuery, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.ref_ids = ref_ids
        self.query_inds = query_inds

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.ref_idxs, self.query_inds, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)

        query_inds = np.loadtxt(os.path.join(self.file_root, self.query_inds), dtype=int)
        ref_idxs = np.loadtxt(os.path.join(self.file_root, self.ref_ids), dtype=int)

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        assert len(img_data) == label_data.shape[0]

        return img_data, label_data, ref_idxs, query_inds, attr_num

    def __len__(self):
        return self.ref_idxs.shape[0]

    def __getitem__(self, index):

        ref_id = int(self.ref_idxs[index])
        img = Image.open(os.path.join(self.img_root_path, self.img_data[ref_id]))
        img = img.convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        indicator = self.query_inds[index]

        return img, indicator


class Data(data.Dataset):
    """
    Load data for attribute predictor training (pre-training)
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_train.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        img_transform: transformation functions for img. Default: ToTensor()
        mode: the mode 'train' or 'test' decides to load training set or test set
    """
    def __init__(self, file_root, img_root_path, img_transform=None, mode='train'):
        super(Data, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)
        assert len(img_data) == label_data.shape[0]

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        return img_data, label_data, attr_num

    def __len__(self):
        return self.label_data.shape[0]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_root_path, self.img_data[index]))
        img = img.convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)

        label_vector = self.label_data[index]  #one-hot

        return img, get_idx_label(label_vector, self.attr_num)


class DataTriplet(data.Dataset):
    """
    Load generated attribute manipulation triplets for training.
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_train.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        triplet_name: the filename of generated txt file, which includes ids of sampled triplets
        mode: 'train' or 'valid'
        ratio: ratio to split train and validation set. Default: 0.9
    """
    def __init__(self, file_root, img_root_path, triplet_name, img_transform=None, mode='train', ratio=0.9):
        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.triplet_name = triplet_name
        self.ratio = ratio

        if self.img_transform is None:
            self.img_transform = transforms.ToTensor()

        self.triplets, self.triplets_inds, self.img_data, self.label_one_hot, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_train.txt")) as f:
            img_data = f.read().splitlines()

        label_one_hot = np.loadtxt(os.path.join(self.file_root, "labels_train.txt"), dtype=int)
        assert len(img_data) == label_one_hot.shape[0]

        with open(os.path.join(self.file_root, "%s.txt" % self.triplet_name)) as f:
            triplets = f.read().splitlines()

        triplets_inds = np.loadtxt(os.path.join(self.file_root, "%s_ind.txt" % self.triplet_name), dtype=int)  #indicators

        N = int(len(triplets) * self.ratio) #split train/val

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        if self.mode == 'train':
            triplets_o = triplets[:N]
            triplets_inds_o = triplets_inds[:N]
        elif self.mode == 'valid':
            triplets_o = triplets[N:]
            triplets_inds_o = triplets_inds[N:]

        return triplets_o, triplets_inds_o, img_data,  label_one_hot, attr_num

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):

        ref_id, pos_id, neg_id = self.triplets[index].split(' ')
        idxs = {'ref': int(ref_id), 'pos': int(pos_id), 'neg': int(neg_id)}
        imgs = {}
        for key in idxs.keys():
            with Image.open(os.path.join(self.img_root_path, self.img_data[idxs[key]])) as img:
                img = img.convert('RGB')
                if self.img_transform:
                    img = self.img_transform(img)
                imgs[key] = img

        one_hots = {}
        for key in idxs.keys():
            one_hots[key] = self.label_one_hot[idxs[key]]

        indicator = self.triplets_inds[index]

        labels = {}
        for key in idxs.keys():
            labels[key] = get_idx_label(one_hots[key], self.attr_num)

        return imgs, \
               one_hots, \
               labels, \
               indicator
