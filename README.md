# DROPOUT CONCRETE AUTOENCODER FOR BAND SELECTION ON HSI SCENES
# PROFESSIONAL  README GENERATOR
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2401.16522)
![Python 3.8](https://img.shields.io/badge/python-3.8.16-green.svg)
![Pytorch 2.1.1](https://img.shields.io/badge/Pytorch-2.1.1-blue.svg)
[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://choosealicense.com/licenses/mit/)

![Network](assets/schema_hyper_img.png)

**Acknowledgment:** This code is mainly based on the works [dl_selection](https://github.com/iancovert/dl-selection.git) and [DiffusionEdge](https://github.com/GuHuangAI/DiffusionEdge.git). 


## I. Training.
./train.sh --cfg configs/xxx.yaml

## II. Prepare Data.
The project folder structure should look like this:
```commandlines
|--myHyper
  |-- $configs
  |   |-- KSC_T1.yaml
  |-- $datasets
  |   |-- KSC_gt.mat
  |   |-- KSC.mat
  |-- data.py
  |-- encoders.py
  |-- concreteVAE.py
  |-- utils.py
  |-- train.py
  |-- train.sh   




