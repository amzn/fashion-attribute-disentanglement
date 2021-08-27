# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch


def get_idx_label(label_vector, attr_num):
    """
    convert the one-hot attribute labels to the label that indicate GT's position for each attribute
    Args:
        label_vector: one-hot labels
        attr_num: 1-D list of numbers of attribute values for each attribute
    """
    start_idx = 0
    labels = []
    for i in attr_num:
        sub_labels = label_vector[start_idx:start_idx + i]
        if sum(sub_labels) > 0:
            assert sum(sub_labels) == 1
            labels.append(np.argmax(sub_labels))
        else:
            labels.append(-1)  # missing label
        start_idx += i
    return np.array(labels)


def get_non_diagonal_elements(attr_num, dim_chunk):
    """
    return the array that indicates the location of non_diagnal elements in the memory block
    Args:
        attr_num: 1-D list of numbers of attribute values for each attribute
        dim_chunk: the dimension of each attribute-specific embeddings
    """
    attr_sum = sum(attr_num)  # number of all attribute values
    vector_dim = len(attr_num) * dim_chunk  # the dimension of each prototype embedding

    # create the block diagonal matrix
    D = torch.zeros(vector_dim, attr_sum)
    start_idx = 0
    for i in range(len(attr_num)):
        D[dim_chunk * i:dim_chunk * i + dim_chunk, start_idx:start_idx + attr_num[i]] = 1
        start_idx += attr_num[i]
    # return non-diagonal parts
    return 1 - D


def split_labels(label, attr_num):
    """
    split the whole one-hot label into separate one-hot labels w.r.t attribute
    """
    labels = []
    start_idx = 0
    for i in attr_num:
        sub_labels = label[start_idx:start_idx + i]
        labels.append(sub_labels)
        start_idx += i
    return labels


def get_target_attr(indicator, attr_num):
    """
    given the indicator vector, return the target attribute that need to be changed
    """
    assert 1 in indicator #sanity check, ensure target attribute exists
    start_idx = 0
    for i, val_num in enumerate(attr_num):
        sub_label = indicator[start_idx:start_idx + val_num]
        if 1 in sub_label:
            return i
        else:
            start_idx += val_num
    return -1


def compute_DCG(scores):
    """
    compute the Discounteed Cumulative Gain. Check the equation in the paper
    """
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def compute_NDCG(rank_scores):
    """
    compute the Normalized Discounteed Cumulative Gain. Check the equation in the paper
    """
    dcg = compute_DCG(rank_scores)
    idcg = compute_DCG(np.ones_like(rank_scores))
    ndcg = dcg / idcg
    return ndcg
