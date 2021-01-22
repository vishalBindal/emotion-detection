# Emotion detection

Implementation of various deep learning models to identify a person's emotional state through an image of his/her facial features.

This was done as an assignment in the course on Machine Learning (COL774). The predictions obtained from the model were used in the kaggle competition https://www.kaggle.com/c/col-774-autumn-2020/overview in which our team *Cypher* placed 4th on the leaderboard.

## Overview

The following models are implemented for the problem:
- A vanilla neural network with a single hidden layer
- A CNN with the following layers: CONV1, POOL1, CONV2, POOL2, FC1, FC2, Softmax
- Various Resnet and Resnext pretrained models in Pytorch

The following are also tested:
- Feature engineering: *Gabor filters* and *Histogram of oriented gradients* from skimage library
- Data augmentation: Augmented training data using horizontally-flipped and rotated versions of the images

More details can be found in the problem statement (*assignment4.pdf*) and report (*final_submission/COL774_Assign4.pdf*)

## Setup

1. Set up a python environment

```
conda create -n env python=3.6
conda activate env
```

2. Install prerequisites

    1. numpy, matplotlib, scikit-learn, scikit-image
    
    
    ```
    conda install numpy matplotlib
    conda install -c conda-forge scikit-learn scikit-image
    ```
    
    2. Pytorch
    
    Follow installation instructions at https://pytorch.org/
    
    e.g. for CUDA 10.2
    
    ```
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    ```
    
3. Download files *train.csv*, *public_test.csv* and *private.csv* from https://www.kaggle.com/c/col-774-autumn-2020/data. Place them in the *datasets* directory.


## Usage

*neural-net.py* contains implementation of vanilla neural network, CNN and feature engineering techniques. *compe_part\*.py* contains implementation of resnet and resnext models, alongwith data augmentation techniques.

To run any file, set appropriate dataset file paths and parameters, and simply run as `python3 <file-name>`  
