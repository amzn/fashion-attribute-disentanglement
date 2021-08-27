# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


def add_base_args(parser):  # common arguments for all CLIs
    parser.add_argument('--dataset_name', type=str,
                        default='Shopping100k',
                        choices=['Shopping100k', 'DeepFashion', 'DARN'],
                        help='Select dataset (Shopping100k or DeepFashion or DARN')
    parser.add_argument('--backbone', type=str,
                        default='alexnet',
                        choices=['alexnet', 'resnet'],
                        help='Select pretrained backbone architecture (alexnet or resnet)')
    parser.add_argument('--file_root', type=str,
                        required=True,
                        help='Path for pre-processed files')
    parser.add_argument('--img_root', type=str,
                        required=True,
                        help='Path for raw images')
    parser.add_argument('--num_threads', type=int,
                        default=16,
                        help='Number of threads for fetching data (default: 16)')
    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--dim_chunk', type=int,
                        default=340,
                        help='Dimension of each attribute-specific embedding')
    parser.add_argument('--load_pretrained_extractor', type=str,
                        default=None,
                        help='Load pretrained weights of disentangled representation learner')
    parser.add_argument('--load_pretrained_memory', type=str,
                        default=None,
                        help='Load pretrained weights of memory block')
    parser.add_argument('--use_cpu', action='store_true',
                        help='Do not use cuda')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Which GPU to use (if any available)')


def add_train_args(parser):
    parser.add_argument('--momentum', type=float,
                        default=0.9,
                        help='Beta1 for Adam optimizer (dafault: 0.9)')
    parser.add_argument('--num_epochs', type=int,
                        default=25,
                        help='Number of epochs (default: 25)')
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.5,
                        help='Learning rate decay (default: 0.5)')
    parser.add_argument('--lr_decay_step', type=int,
                        default=10,
                        help='Learning rate decay step (default=10)')
    parser.add_argument('--triplet_file', type=str,
                        default='triplet_train',
                        help='name of off-line generated triplet file')
    parser.add_argument('--diagonal', action='store_true',
                        help='Add diagonal regularization to loss')
    parser.add_argument('--consist_loss', action='store_true',
                        help='Add consistency loss')
    parser.add_argument('--weight_cls', type=float, default=0.2,
                        help='Weight for classification losses (cross-entropy)')
    parser.add_argument('--weight_label_trip', type=float, default=1.0,
                        help='weight for label triplet loss')
    parser.add_argument('--weight_manip_trip', type=float, default=1.0,
                        help='Weight for manipulation triplet loss')
    parser.add_argument('--weight_consist', type=float, default=1.0,
                        help='Weight for consistency loss')
    parser.add_argument('--weight_diag', type=float, default=0.4,
                        help='Weight for diagonal regularization')
    parser.add_argument('--weight_label_trip_local', type=float, default=0.2,
                        help='Weight for label triplet loss for each chunk')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Margin for rank loss')
    parser.add_argument('--load_init_mem', type=str, default=None,
                        help='Load initial memory block generated from init_mem.py')
    parser.add_argument('--ckpt_dir', type=str,
                        default=None,
                        help='Directory to save model weights (default: use timestamp)')


def add_eval_args(parser):
    parser.add_argument('--ref_ids', type=str,
                        default='ref_test.txt',
                        help='list of query image id')
    parser.add_argument('--gt_labels', type=str,
                        default='gt_test.txt',
                        help='list of target labels')
    parser.add_argument('--query_inds', type=str,
                        default='indfull_test.txt',
                        help='list of indicators')
    parser.add_argument('--top_k', type=int,
                        default=30,
                        help='top K neighbours')
    parser.add_argument('--save_matrix', action='store_true',
                        help='Save the gallery feature and fused feature')
    parser.add_argument('--feat_dir', type=str,
                        default='eval_out',
                        help='Path to store gallery feature and fused feature')


def add_init_args(parser):
    parser.add_argument('--memory_dir', type=str,
                        required=True,
                        help='Directory to save initialized memory block')
    parser.add_argument('--num_sample', type=int,
                        default=100,
                        help='Number of image samples for each attribute value')
