# Learning Attribute-driven Disentangled Representations for Interactive Fashion Retrieval

This code implements the Attribute-Driven Disentangled Encoder (ADDE) as well as the model for
attribute manipulation (ADDE-M) as described in the original manuscript:

> Yuxin Hou, Eleonora Vig, Michael Donoser, and Loris Bazzani. [Learning Attribute-driven Disentangled Representations for Interactive Fashion Retrieval](https://www.amazon.science/publications/learning-attribute-driven-disentangled-representations-for-interactive-fashion-retrieval). ICCV 2021.

If you find this code useful in your research, please cite the following:

```
@inproceedings{hou2021disentanglement,
    title={Learning Attribute-driven Disentangled Representations for Interactive Fashion Retrieval},
    author={Hou, Yuxin and Vig, Eleonora and Donoser, Michael and Bazzani, Loris},
    booktitle = {The International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2021}
}
```

Note: the implementation of this work was mainly done by Yuxin Hou during her internship at Amazon.

## Before Cloning: git-lfs

Install [git-lfs](https://git-lfs.github.com/) before cloning by following the instructions [here](https://github.com/git-lfs/git-lfs/wiki/Installation).
This repository uses git-lfs to store model checkpoint and split files.

Once installed, the files in git-lfs will be automatically downloaded when cloning the repository with:
```
git clone https://github.com/amzn/fashion-attribute-disentanglement.git
```

These files can optionally be ignored (since the models are ~500MB) by using `git lfs install --skip-smudge` before cloning the repository, and can be downloaded at any time using `git lfs pull`.


## Installation 

Follow the [instructions](https://www.anaconda.com/products/individual) to install Anaconda. 
You can use python venv if preferred.

```bash
conda create -n adde-m python=3.7
conda activate adde-m
conda install --file requirements.txt
pip install faiss-cpu==1.6.3 torchvision==0.5.0
```

## Data Preparation
We use the following publicly-available datasets that you will need to download from the original sources:
+ **Shopping100k**: contact [the author of the dataset](https://sites.google.com/view/kenanemirak/home) to get access to the images.
+ **DeepFashion**: download images and labels for the category and attribute prediction benchmark from [the dataset website](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html).

In the sub-folders `models` and `splits`, you should be able to find all the material needed to run training and evaluation.
Note: this folders were downloaded via git-lfs; if they are not present, please makes sure that git-lfs is installed **before cloning** the git repository. 


## Training

The root path of the dataset you want to train on should be exported like this:
```bash
export DATASET_PATH="/path/to/dataset/folder/that/contain/img/subfolder"
```
And you can choose which dataset to use with:
```bash
export DATASET_NAME="Shopping100k"  # Shopping100k or DeepFashion
```
And provide the path where to store the models:
```bash
export MODELS_DIR="/path/to/saved/model/checkpoints"
```

There are 3 different stages that should be run sequentially.

Train the attribute-driven disentangled encoder for attribute predictors (ADDE):
```
python src/train_attr_pred.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} --img_root ${DATASET_PATH} --ckpt_dir ${MODELS_DIR}/encoder --batch_size 128
```
After training finished, inspect the log file `${MODELS_DIR}/encoder/log.txt` to find the best model checkpoint accordingly to the validation accuracy. 
The best model is automatically stored in `${MODELS_DIR}/encoder/extractor_best.pkl`.
 
Initialize the memory block:
```
python src/init_mem.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} --img_root ${DATASET_PATH} --load_pretrained_extractor ${MODELS_DIR}/encoder/extractor_best.pkl --memory_dir ${MODELS_DIR}/initialized_memory_block
```
The output is stored in `${MODELS_DIR}/initialized_memory_block`.

Train the attribute manipulation model (ADDE-M):
```
python src/train_attr_manip.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} --img_root ${DATASET_PATH}  --ckpt_dir ${MODELS_DIR}/checkpoints --load_pretrained_extractor ${MODELS_DIR}/encoder/extractor_best.pkl --load_init_mem ${MODELS_DIR}/initialized_memory_block/init_mem.npy --diagonal --consist_loss 
```
The output is stored in `${MODELS_DIR}/checkpoints`.


## Evaluation

Once the ADDE-M model is trained, you can run the evaluation on the attribute manipulation task as follows:
```
python src/eval.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} --img_root ${DATASET_PATH} --load_pretrained_extractor ${MODELS_DIR}/checkpoints/extractor_best.pkl --load_pretrained_memory ${MODELS_DIR}/checkpoints/memory_best.pkl 
```
This produces top-30 accuracy and NCDG. Add `--top k` command to get results for different values of top-k.


## Pretrained models

We provide pretrained model weights for ADDE-M (attribute manipulation) under the `models` directory. 

To get the top-30 accuracy and NCDG of the paper, you can run:
```
export MODELS_DIR="./models/Shopping100k";
python src/eval.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} --img_root ${DATASET_PATH} --load_pretrained_extractor ${MODELS_DIR}/extractor_best.pkl --load_pretrained_memory ${MODELS_DIR}/memory_best.pkl 
```

To further finetune the pretrained models, you can run:
```
python src/train_attr_manip.py --dataset_name ${DATASET_NAME} --file_root splits/${DATASET_NAME} --img_root ${DATASET_PATH} --ckpt_dir ${MODELS_DIR}/checkpoints_finetune/ --load_pretrained_extractor ${MODELS_DIR}/extractor_best.pkl --load_pretrained_memory ${MODELS_DIR}/memory_best.pkl --diagonal 
```
In order to perform finetuning on a different dataset than the supported ones, one would need to provide support files as described in the next section. 


## Preparing Support Files

We describe here how the files in the `splits` folder were prepared so one can run experiments on a different dataset.

### Files for Training

The dataset need to be pre-processed to generate train/test split set, triplets and queries. 
These are the files that one would need to create:
+ `imgs_train.txt`,`imgs_test.txt`: they store the name of images in relative path for training and testing.
+ `labels_train.txt`,`labels_test.txt`: they store one-hot attribute labels, used to train the attribute-driven disentangled encoder. 
`labels_*.txt` is paired with `imgs_*.txt`, that is, the i-th line in  `labels_*.txt`  is the label vector for the i-th line image in `img_*.txt`.
+ `attr_num.txt`: a list that consists of the number of attribute values for each attribute type.

   For example in Shopping100k, we list is `[16, 17, 19, 14, 10, 15, 2, 11, 16, 7, 9, 15]` since there are 12 attribute types in total, and the first attribute (category) has 16 values, the second has 17 and so on.

The triplets for attribute manipulation are generated offline.
For each triplet, we create the indicator vector by randomly selecting an attribute type and a new attribute value for the attribute.
Then based on the indicator vector, we randomly pick the images with all target attributes as positive samples, 
and randomly pick images that have different attributes as negative samples.  We should finally generate the two paired files:
+ `triplet_train.txt`: each line corresponds to a triplet. A triplet is defined by 3 space-separated indexes (reference, positive, negative). The index of images is the line index of the file `labels_train.txt`.

   E.g. each line should be `reference_id positive_id  negative_id`
+ `triplet_train_ind.txt`: each line is the indicator vector for each triplets. Each number in a line is 1, 0, or -1 if the attribute is changed but in the reference, preserved or changed but in the target image, respectively.

### Files for Evaluation Queries

We enumerate all attribute types and all different attribute values to generate target attributes,
and if there is image has the target attributes, the query is valid. We generate the following paired files:
+ `ref_test.txt`: 1-D list that consists of indexes (line index in `labels_test.txt`) of reference images.
+ `indfull_test.txt`: array that consists of the full indicator vector (same dimension as the merged one-hot label vector) for each reference image.
+ `gt_test.txt`: array that stores ground truth target labels for each query.

For the i-th query, the i-th line in `ref_test.txt` indicates the index of reference images in the test set, 
the i-th line in `indfull_test.txt` is the indicate vector consists of -1, 0, 1,
and the i-th line in `gt_test.txt` is the one-hot label vectors of the target attributes.


## Security

See [CONTRIBUTING](https://github.com/amzn/fashion-attribute-disentanglement/blob/main/CONTRIBUTING.md#security-issue-notifications) for more information.


## License

This library is licensed under the CC-BY-NC-4.0 License.
