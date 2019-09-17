# Automated Spectral Kernel Learning
## Intro
This repository provides the code used to run the experiments of the paper "Automated Spectral Kernel Learning" (https://arxiv.org/abs/1909.04894).
## Environments
- Python 3.7.4
- Pytorch 1.2.0
- CUDA 10.1.168
- cuDnn 7.6.0
- GPU: Tesla P100 16GB
## Core functions
- auto_kernel_learning.py implements the algorithm to construct an one-layer neural network, including initialization of trainable weights and untrainable biases as well as feature mapping (cosine as activation).
- utils.py implements useful tools including load svmlight style dataset and classic datasets used in Pytorch but also various loss functions are introduced.
- optimal_parameters.py records optimal parameters for the proposed algorithm.
## Experiments
1. Download datasets for multi-class classification (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
2. Run the script to tune parameters and record them in optimal_parameters.py.
```
python run_parameter_tune.py
```
3. Run the script to obtain results in Experiment section
```
python run_exp1.py
```
