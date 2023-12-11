# Isotropic Representation Can Improve Dense Retrieval
This github repository is for the paper "Isotropic Representation Can Improve Dense Retrieval" published at PAKDD 2023.

## Package
We used packages listed below.
```
python=3.9.15
pytorch=1.13.1
transformers=4.12.5
pytools=2022.1.13
isoscore=1.0
```

## Libraries
Thanks to github libraries, [glow-pytorch](https://github.com/rosinality/glow-pytorch) and [nice_pytorch](https://github.com/gmum/nice_pytorch), I implemented Glow and NICE.   
Clone these libraries, then you can use the clodes for Glow and NICE.

## Dataset
We used Robust04b, ClueWeb09b, and MS-MARCO.   
You can download the datasets we used [here](https://drive.google.com/drive/u/1/folders/1f8zJ61L7t4DzGnDqNKbHBykwoLADg7Az).   
In the experiments, we use the name 'robust', 'wt' and 'msmarco' for Robust04, ClueWeb09b, and MS-MARCO respectively.   
Robust04 and ClueWeb09b datasets are both divided into five folds, while MS-MARCO has only one fold.   
When you train the model on MS-MARCO, you should set the argument 'msmarco' as True, and you can limit the train data size by setting the argument 'batches_per_epoch'.   
We used 1024 as 'batches_per_epoch' when training the models using MS-MARCO.

## Model
We adopted BERT (a pre-trained model named 'bert-base-uncased' provided by huggingface) as a backbone model.   
We implemented BERT-based neural ranking models (ColBERT and RepBERT) and trained them.

## Running the codes
We provide bash files to run the codes.   
To train the models using the dataset of MS-MARCO, you should run the bash files where 'msmarco' is attached at the end of the filename. (We explain how to run files based on the Robust09 and ClueWeb09b here.)
When applying normalizing flow or whitening to RepBERT, you can choose whether to transform representations **token-wise** (tw) or **sequence-wise** (sw).
1. You should fine-tune ColBERT or RepBERT using *run_ft_colbert.sh* or *run_ft_repbert.sh*
2. When the fine-tuning is done, you can train normalizing flow models or compute mean and standard deviation vectors for whitening. For running NICE, you run *run_nice_colbert.sh*, *run_nice_tw_repbert.sh*, or *run_nice_sw_repbert.sh*. For running Glow, you run *run_glow_colbert.sh*, *run_glow_tw_repbert.sh*, or *run_glow_sw_repbert.sh*. To apply whitening, you use the files *run_whitening_colbert.sh* and *run_whitening_repbert.sh*.