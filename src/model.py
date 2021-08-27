# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch.nn as nn
import torchvision


class Extractor(nn.Module):
    """
    Extract attribute-specific embeddings and add attribute predictor for each.
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        backbone: String that indicate the name of pretrained backbone
        dim_chunk: int, the size of each attribute-specific embedding
    """
    def __init__(self, attr_nums, backbone='alexnet', dim_chunk=340):
        super(Extractor, self).__init__()

        self.attr_nums = attr_nums
        if backbone == 'alexnet':
            self.backbone = torchvision.models.alexnet(pretrained=True)
            self.backbone.classifier = self.backbone.classifier[:-2]
            dim_init = 4096
        if backbone == 'resnet':
            self.backbone = torchvision.models.resnet18(pretrained=True)
            self.backbone.fc = nn.Sequential()
            dim_init = 512

        dis_proj = []
        for i in range(len(attr_nums)):
            dis_proj.append(nn.Sequential(
                    nn.Linear(dim_init, dim_chunk),
                    nn.ReLU(),
                    nn.Linear(dim_chunk, dim_chunk)
                )
            )
        self.dis_proj = nn.ModuleList(dis_proj)

        attr_classifier = []
        for i in range(len(attr_nums)):
            attr_classifier.append(nn.Sequential(
                nn.Linear(dim_chunk, attr_nums[i]))
            )
        self.attr_classifier = nn.ModuleList(attr_classifier)

    def forward(self, img):
        """
        Returns:
            dis_feat: a list of extracted attribute-specific embeddings
            attr_classification_out: a list of classification prediction results for each attribute
        """
        feat = self.backbone(img)
        dis_feat = []
        for layer in self.dis_proj:
            dis_feat.append(layer(feat))

        attr_classification_out = []
        for i, layer in enumerate(self.attr_classifier):
            attr_classification_out.append(layer(dis_feat[i]).squeeze())
        return dis_feat, attr_classification_out


class MemoryBlock(nn.Module):
    """
    Store the prototype embeddings of all attribute values
    Args:
        attr_nums: 1-D list of numbers of attribute values for each attribute
        dim_chunk: int, the size of each attribute-specific embedding
    """
    def __init__(self, attr_nums, dim_chunk = 340):
        super(MemoryBlock, self).__init__()
        self.Memory = nn.Linear(np.sum(attr_nums), len(attr_nums) * dim_chunk, bias=False)

    def forward(self, indicator):
        t = self.Memory(indicator)
        return t
