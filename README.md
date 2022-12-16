# DCAT

This repository is the implementation of *Dual-branch Cross-Patch Attention Learning for Group Affect Recognition*. [ArXiv](https://arxiv.org/abs/2212.07055)

## Installation

To install requirements:

```setup
pip install -r requirements.txt
```

With conda:

```
conda create -n crossvit python=3.8
conda activate crossvit
conda install pytorch=1.7.1 torchvision  cudatoolkit=11.0 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Get MIP

```
pip install mtcnn
pip install tensorflow
cd GetMIP_POINT/
```

Edit GAF_Face_filter.py line.27-29, to fill in the original image path, the directory to save your preprocess images, the path to save the index of preprocess images.

To get preprocess images, run:
```shell script
python GAF_Face_filter.py
```

Create a new conda environment with the requirements below:
```
(1) Pytorch 1.0.0
(2) Python 3.6+
(3) Python packages: numpy, scipy, pyyaml/yaml, h5py, opencv, PIL
```

Download the pretrained model of POINT from: https://github.com/harlanhong/POINT

To get MIP result, run:
```shell script
python POINT_new_dataset_test.py \
--index_name [GAF_Face_filter.py line.29] \
--dataset_path [GAF_Face_filter.py line.28] \
--result_dir [...] \
--model [path to pretrained model of POINT] --h 4 --N 2
```
e.g.
```shell script
python POINT_new_dataset_test.py \
--index_name ./data/GAF3_process/Validation/Neg_index.npy \
--dataset_path ./data/GAF3_process/Validation/Negative \
--result_dir ./resultFile/GAF3/Validation/Negative/ \
--model ./models/MS_h4_N2.pkl --h 4 --N 2
```




## Pretrained models
We provide models trained on GAF 3.0 and GroupEmoW. You can find models [here](https://drive.google.com/drive/folders/1-qFKcqDHCaGP1HIMfueXZTlrZ12O1scA?usp=sharing).




## Training

To train `DCAT` on GAF3.0 on a single node with 1 gpus for 300 epochs run:

```shell script
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 --use_env main.py --batch-size 64 \
--data-path /path/to/GAF_3.0/ \
--mip_root_train /path/to/GAF_3.0/train/ \
--mip_cropped_root_train /path/to/GAF3_process/Train/ \
--mip_txt_root_train /path/to/resultFile/GAF3/Train/ \
--mip_root_val /path/to/GAF_3.0/val/ \
--mip_cropped_root_val/path/to/GAF3_process/Validation/ \
--mip_txt_root_val /path/to/resultFile/GAF3/Validation/ \
--output_dir ./checkpoint --data-set GAF --mip_select --mip_keep_ratio 0.5 --full_select --full_keep_ratio 0.5
```


## Evaluation

To evaluate a pretrained model on `DCAT`:

```shell script
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 --use_env main.py --batch-size 64 \
--data-path /path/to/GAF_3.0/ \
--mip_root_train /path/to/GAF_3.0/train/ \
--mip_cropped_root_train /path/to/GAF3_process/Train/ \
--mip_txt_root_train /path/to/resultFile/GAF3/Train/ \
--mip_root_val /path/to/GAF_3.0/val/ \
--mip_cropped_root_val/path/to/GAF3_process/Validation/ \
--mip_txt_root_val /path/to/resultFile/GAF3/Validation/ \
--output_dir ./checkpoint --eval --data-set GAF --mip_select --mip_keep_ratio 0.5 --full_select --full_keep_ratio 0.5 \
--load_pretrained /path/to/pretrained_models/GAF3/model_best.pth
```
