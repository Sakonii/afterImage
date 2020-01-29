# Inpaint Network

An attempt to assist photo editing tasks using Deep Learning techniques.
This repo remains un-updated till release of fastai-v2.

## Requirements and dependencies

``` bash
# Python3.7
apt-get install python3.7-dev

# python3.7-pip
sudo apt-get install python3-pip
python3.7 -m pip install pip

# Pytorch
sudo -H python3.7 -m pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# For GPU or other version: https://pytorch.org/get-started/locally/

# FastAI
python3.7 -m pip install --user fastai

# Pillow-5.1.0
python3.7 -m pip install --user -U Pillow

# OpenCV-4.1.1.26
python3.7 -m pip install --user opencv-python

# Jupyter
python3.7 -m pip install --user jupyterlab
sudo apt install jupyter-notebook
```

## Model

``` bash
# Download trained models to ./models directory
https://drive.google.com/open?id=1-1GmGvRSU_bZbWG604y_Uuz3hVU8YTKD
https://drive.google.com/open?id=1h45VaLNWvy9WtN7cvFkm94ihWCBriK49

## Inference

``` bash
# Example:
python3.7 main.py --image img.png

# CLI Arguments:
* '--image' : Filename of input image located at img_input directory
```
