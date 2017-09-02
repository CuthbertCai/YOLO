## Tensorflow-YOLO ##

This is the implementation of [YOLO][1] referring to the nilboy's [tensorflow-yolo][2]. This version is for tensorflow-
1.3 and python 3.5, some APIs in this version are different from nilboy's  implementation. In this work, I made some
changes in the data input pipeline and the solver of the network comparing with nilboy's project. In addtion, according
to my experiments, the yolo_net are not able to be trained without the pretraining. So I recommend using the pretraining
ckpt file yolo_tiny.ckpt which you'll see how to download in the following sections and yolo_tiny_net for experiments.

### Require ###
```
Tensorflow-1.3
Python 3.5
Numpy 1.12.0
Opencv-python 3.3.0.9
```
### Download pretrained model ###
[yolo_tiny.ckpt][3]  
`mv yolo_tiny.ckpt models/pretrain`

### Train ###
#### Train on pascal-voc2007 data ####
##### Download pascal-voc2007 data #####
1. Download the training, validation and test data
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval-06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest-06-Nov-2007.tar
```
2. Extract all of these tars into one directory named VOCdevkit
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.ta
```
3. It should have this basic structure
```
$VOCdevkit/
$VOCdevkit/VOC2007
```
4. Create symlinks for the PASCAL VOC dataset
```
cd $YOLO_ROOT/data
ln -s $VOCdevkit VOCdevkit2007
```
Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple
projects

##### convert the PASCAL-voc data to text_record file #####
`python tools/preprocess_pascal_voc.py`

##### train #####
`python tools/train.py -c conf/train.cfg`

### Test demo ###
`python demo.py`

### Notice ###
In the demo, only one object can be detected for each image.

[1]:https://arxiv.org/pdf/1506.02640.pdf
[2]:https://github.com/nilboy/tensorflow-yolo
[3]:https://drive.google.com/file/d/0B-yiAeTLLamRekxqVE01Yi1RRlk/view?usp=sharing
