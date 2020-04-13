# afterImage

An attempt to assist photo editing tasks using Deep Learning. [Under Construction]

## Requirements and dependencies

Python3.7 and pip

```bash

# Python3
apt-get install python3.7-dev

# python3.7-pip
sudo apt-get install python3-pip
python3.7 -m pip install pip
```

Dependencies installation under pip package manager

``` bash

# Pytorch
sudo -H python3.7 -m pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# For GPU or other version: https://pytorch.org/get-started/locally/

# OpenCV-4.1.1.26
python3.7 -m pip install --user opencv-python

# Jupyter
python3.7 -m pip install --user jupyterlab
sudo apt install jupyter-notebook
```

Build detectron2 from source:

``` bash
# Detectron2-0.1.1
https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

## Inference

``` bash
# Example:
python3.7 main.py --image img.png

# CLI Arguments:
* '--image' : Filename of input image located at img_input/ directory
* '--model_segmentation' : Filename of weights associated with segmentation
* '--cfg_path' : Path to model cfg file relative to 'detectron2/model_zoo/configs'
* '--img_path' : Custom path for input image
```

## Results

* Not Available
