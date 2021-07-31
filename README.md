# Research and Application On Face Super Resolution Algorithm Based On Sub-pixel Convolution and Residual Learning(FSRSR)
This repository contains the implementation for FSRSR
## Environment
- python 3.8
- pytorch 1.8.1
- GPU : NVIDIA Geforce 940MX
- miniconda, numpy, tqdm, cv2, visdom, tkinter
## Dataset
### TrainSet/ValueSet
CelebA:from [official BaiduNetdisk link](https://pan.baidu.com/s/1eSNpdRG#list/path=/) or [official site](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
### TestSet
CelebAHQ:[official site](https://paperswithcode.com/dataset/celeba-hq)
LFW:[official site](http://vis-www.cs.umass.edu/lfw/)
SeePrettyFace:[official site](http://www.seeprettyface.com/)
## Preparation For LR-HR pairs
generate LR-HR pairs from your train set, value set, and test set
```
python data_utils.py
```
## Train
use visdom to visualize training process
```
python -m visdom.server
```
open another terminal to start training
```
python train.py
```
the results will be stored in folder:/results, the trained model is stored in folder:/epochs

