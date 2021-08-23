# Research and Application On Face Super Resolution Algorithm Based On Sub-pixel Convolution and Residual Learning(FSRSR)
This repository contains the implementation for FSRSR and its application

## Research
---
### Environment
- python 3.8
- pytorch 1.8.1
- GPU : NVIDIA Geforce 940MX
- miniconda, numpy, tqdm, cv2, visdom, tkinter
### Dataset

Train Set|Link
--|:--:
CelebA|<a href="https://pan.baidu.com/s/1eSNpdRG#list/path=/" target="_blank">official BaiduNetdisk link</a> or <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">official site</a>

Test Set|Link
--|:--:
CelebAHQ|<a href="https://paperswithcode.com/dataset/celeba-hq" target="_blank">official site</a>
LFW|<a href="http://vis-www.cs.umass.edu/lfw/" target="_blank">official site</a>
SeePrettyFace|<a href="http://www.seeprettyface.com/" target="_blank">official site</a>


### Preparation For LR-HR pairs
generate LR-HR pairs from your train set, value set, and test set
```
python data_utils.py
```
### Train
use visdom to visualize training process
```
python -m visdom.server
```
open another terminal to start training
```
python train.py
```
the results will be saved in folder: `/results`, the trained model is stored in folder: `/epochs`

### Test
then you can use the trained model to test an image or a video by:
```
python test_image.py
```

```
python test_video.py
```

## Application
---
### Run
the application of FSRSR algorithm is coded in python with <a href="http://tcl.tk/man/tcl8.5/TkCmd/contents.htm" target="_blank">tkinter</a> as an interface, to run it by:
```
python MainWindow.py
```
here I apply this algorithm from two aspects : **Real Face Reconstruction** and **Local Dataset Reconstruction**

in `Real Face Reconstruction`, you can call local computer camera to get a Real Face image as LR, then use pretrained FSRSR model to generate a SR image, finally the DNN Face Detection function will be called to compare the quality of LR and HR images by drawing a rectangle of features with its confidence.

in `Local Dataset Reconstruction`, you can select a face picture from local dataset, which has paired LR-HR images, the system can automatically calculate SSIM and PSNR of LR image, and three algorithms APIs are provided(FSRSR, FSRCNN, ESPCN) to generate the corresponding SR image, whose SSIM and PSNR are also be calculated for compare. 

**Notes**: Three models need to be pretrained by your own dataset before called in this system.

