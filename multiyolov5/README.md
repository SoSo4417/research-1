# Multi YOLO V5——Detection and Semantic Segmentation
## Overeview
This is a part of my master project which based on <a href="https://github.com/ultralytics/yolov5"> ultralytics YOLO V5 tag v5.0</a>.  
This multi-task model adds only a small amount of computation and inferential GPU memory (about 350MB) and is able to accomplish both object detection and semantic segmentation. Object detection metrics are slightly improved (compared to single-task YOLO) on my dataset (transfer from Cityscapes Instance Segmentation labels) and Cityscapes semantic segmentation metrics are shown below.

test video:https://youtu.be/dLjktV2vsF0


![40](https://user-images.githubusercontent.com/98376235/156918119-2c03820b-1f7f-4459-a9cd-1e64c1779dc0.png)
![43](https://user-images.githubusercontent.com/98376235/156918124-e3031fa8-bffc-42b1-b8b8-19b1608c3d41.jpg)

![44](https://user-images.githubusercontent.com/98376235/156918126-fc43cdfc-1bed-49fd-8a43-817c05d7c7c2.png)

Doc

0. Before Start Environments Configuration and Dataset Preparation
(a) Environment
For now it is recommended to use the main branch BS2021 directly, no more updates to features and structures will be made in the near future, but if issues raise bugs we will try to fix them in the main branch when we are free

$ python -m pip install -r requirements.txt  
$ python -m pip uninstall wandb  
Note! The current code doesn't do multi-card training and wandb support, so there is a high probability of bugs without uninstalling training

(b) Dataset Prepare
The model detection and segmentation datasets are loaded independently and do not require the same category. The current support Cityscapes semantic segmentation dataset and instance segmentation tag generation target detection dataset (new BDD100k mixed Cityscapes training support, BDD100k as a cityscapes a city organization dataset format), extended semantic segmentation dataset need to add and change the code, inherit BaseDataset class. The target detection dataset can be replaced with the original yolo by itself, refer to the original YOLOV5 documentation and. /data/cityscapes_det.yaml file
Download data: Download the Cityscapes dataset from the official website, put leftImg8bit and gtFine into. /data/citys folder, you can also use the bash script in cityscapes to download, you need to change the account password in the script to your own first
Data preprocessing: Go to the citys folder and run 2yolo_filter.py in the citys folder to generate the target detection tags. Create a new detdata folder in the citys folder, and cut the generated images and labels folders to detdata.
Note: more space-consuming, only Cityscapes experiments can be considered to delete the copied images, the leftImg8bit figure soft link to detdata (but do not move the leftImg8bit and gtFine folder, segmentation to be used)

$ cd . /data/citys
$ python 2yolo_filter.py
$ mkdir detdata
$ mv . /images . /detdata
$ mv . /labels . /detdata

Version 2.0 prepares 4 pre-trained models for splitting Head. From the visualization point of view, we recommend psp and lab (the feeling field is bigger), there is no big difference in speed, base is the fastest, psp is the second, lab and bise are close.
Recommended index: Lab and PSP > Base and BiSe
base.pt base version of the segmentation head. 16 layers (PAN1/8) input, profile channel 512. C3, channel slightly widened version of C3SPP, dropout (0.1), 1 × 1 convolution to the category. Speed accuracy integrated effect is good, but SPP with 1/8 figure feeling field is actually not large enough, s model is good enough, but m model deepening and widening after the amount of improvement is not satisfactory.
bise.pt mimics BiSeNetV1's splitting head, with slightly larger accuracy and speed similar to base. 16,19,22 (1/8,1/16,1/32 of PAN) input, profile channel invalid. ARM changed to RFB2 enhanced nonlinear BiSeNet has a 3×3 convolutional refine after each Upsample, here save calculation in front of the Upsample. The auxiliary loss coefficient of BiSeNet is 1, here the auxiliary loss is too large for bad results.
lab.pt mimics DeepLabV3+ segmentation head, validation set accuracy close to psp and bise, slightly slower than psp and base, similar to bise. 4(or 3),19(shallow 1/8,1/16 of PAN) inputs, profile channel 256. 1/8 map 1×1 convolution to 48 channels, 1/16 map over RFB1 (ASPP DeepLabV3+ decoder part uses shallow 1/4 and deep 1/16, here is 1/8 and 1/16 because YOLO 1/4 map channel number is too small and too shallow, after parallel not 3×3refine will be more broken, refine is too computationally intensive. The paper mentions that the shallow large resolution map with fewer channels is more conducive to training, the same paper to 48. The paper mentions that VOC is better with ASPP global, Cityscapes is worse with global, global is not used here (experiments with global edge will be more broken, psp is better with global). Compared with DeepLab decoder part, here more FFM attention fusion structure, in order to use 3×3 cut a little bit of hidden layer to reduce the amount of computation.
psp.pt imitates the split head of PSPNet, which has the highest accuracy and is second only to base in speed. 16,19,22 three layers of fusion input, no suitable place to put the auxiliary loss, and give up the auxiliary loss

pspv5s.pt means psp head yolov5s model, pspv5m.pt means yolov5m several other named the same, pre-trained model is mostly trained with the above cityscapes segmentation data and instance segmentation generated detection data, 19 segmentation classes, 10 detection classes. pspv5m_citybdd_ conewaterbarrier.pt The segmentation part of this model uses a mixture of bdd100k and cityscapes two datasets, the detection part of the data is not open, all kinds of vehicles are classified as vehicle, pedestrain and rider are classified as person, bike and motorcycle are classified as cycle, and another triangle cone cone and water horse waterbarrier category.

1. Inference Inference image, video, video with continuous frames, submitted to Cityscapes, speed measurement
(a) General image inference
$ python detect.py --weights . /pspv5s.pt or other models --source data/images --conf 0.25 --img-size 1024  
There are several images in data/images from the cityscapes test set, bdd100k, apollo scape and yolo. in addition, data/test_imgs puts some apollo plots, so you can see the effect of cityscapes training on apollo (the effect of bdd100k training will be a little better)
The result images are in the runs/detect folder, which can also be inferred and displayed at the same time.

$ python detect.py --weights . /pspv5s.pt or other models --source data/images --conf 0.25 --img-size 1024 --view-img  
Same as the original YOLOV5, --weights writes your pt file, --source writes the path to the image folder or video file, --conf detects the threshold, --img-size is the target long side size of the resize to model

(b) same size continuous frame image to make video
$ python detect.py --weights . /pspv5s.pt or other models --source image folder --conf 0.25 --img-size 1024 --save-as-video  
I only wrote the same size images to create video support (for example, Cityscapes provides three continuous frame test images, bilibili's demo video is these images), put your images into the same folder, note that if there are different size images then the result video will fail to save, if you open --no-save video save the image will not draw the result (do not open)

(c) Submit test set results to Cityscapes
$ python detect.py --weights . /pspv5s.pt or other model --source image folder --conf 0.25 --img-size 1024 --submit --no-save  
If you turn on --no-save and don't save the results, it will be much faster and save space. Combine the images of the 6 folders of the test set in one folder for inference, and you will find a results folder in the runs/detect/this exp, which is the result of converting trainid to id, compress it and upload it to the official website.
You can also reason about the 6 folders separately, and the results will be compressed and uploaded

(d) Speed measurement
Speed measurement will use the parameters submitted in (c) to measure on the same size image, or reasoning video measurement. (a) the picture reasoning is not open cudnn.benchmark, reasoning video files or open --submit or open --save-as-video will open cudnn.benchmark, at this time is the real running speed
Note: cudnn.benchmark will test various cudnn operators and record them on the first frame after cudnn.benchmark is turned on, then the fastest operator will be used on every frame of the same size. cudnn.benchmark will only be turned on when reasoning on the same size image, otherwise it will be measured once every time a new size image is input.
By default, yolov5 uses float16 inference, which is not very different between 20 and 30 series cards, but it will be much slower on 10 and 16 series cards without cudnn.benchmark, so it is recommended to measure the speed when cudnn.benchmark is on.

2. test the model after training

$ python test.py --data cityscapes_det.yaml --segdata . /data/citys --weights . /pspv5s.pt --img-size 1024 --base-size 1024

Two more parameters than the original version: --segdata followed by the Cityscapes dataset folder address (only this is supported now, you can extend it yourself by referring to SegmentationDataset.py)
The parameters for detecting long edges and segmenting long edges are separated, --img-size is to detect long edges --base-size is to segment long edges, my configuration is to put Cityscapes in 1024*512 size inference, more can take into account the speed and accuracy, training is also used for the purpose of tuning the parameters.
If you test your own dataset after training, use test_custom.py (train_custom.py will measure during training)

$ python test_custom.py --data your .yaml --segdata your segmentation data path --weights . /pspv5s.pt --img-size 1024 --base-size 1024

3. Train How to reproduce my results
Before training, download the corresponding original (note that I changed it from the tag V5.0 code) COCO pre-training model to do the initialization, see the original readme and weights/download_weights.sh script

$ python train.py --data cityscapes_det.yaml --cfg yolov5s_city_seg.yaml --batch-size 18 --epochs 200 --weights . /yolov5s.pt --workers 8 --label-smoothing 0.1 --img-size 832 --noautoanchor

It is not necessary to train 200 rounds as in the example (this is the parameter I trained the above pre-trained model to make it converge as much as possible), it is recommended to train at least 80 rounds, I usually train 150 to 180 rounds
The above mentioned my target long edge is 1024, but here is 832, this version of the code in order to save memory to increase the batchsize and facilitate the attempt to add aux loss decided to train on 832 tuning parameters, 1024 on the inference. Training in the output of the detection indicator is 832, the segmentation indicator is 1024, it is recommended that the training and then test.py test the results of 1024
With --noautoanchor because COCO's anchor is just right for cityscapes1024 input (832 autoanchor is small), can alleviate the problems on the anchor. Even so 832 on the training 1024 inference although the indicator is high, but the visualization will see some anchor problems. If your graphics card has 11G, you can appropriately reduce the batchsize directly with 1024 to train
Note: In order to speed up the training I set every 10 rounds to test the segmentation accuracy, the last 40 rounds of each round to test the segmentation accuracy
Be sure to ensure that during the warmup (that is, before I print the accumulate to reach the target value) loss does not occur too large oscillation (phenomenon: the appearance of Nan, the loss of running away, seriously affecting the detection of cls loss. A round to two rounds of split detection loss goes high immediately back down is normal), the above phenomenon consider cutting the learning rate, the current learning rate theoretically all kinds of batchsize should not run away.
The current learning rate theoretically various batchsize should not run away.

4. Code Guide I modified what, adjust the reference / improve the guide
Gradient accumulation
The learning rate and the detection segmentation loss ratio (the latter is not exposed in train.py) are a very important set of parameters. It must be clear that YOLOV5 uses gradient accumulation, regardless of your batchsize, the "nominal batchsize" is the author's preset of 64. This means that when you set the batchsize to 16, the parameters will only be updated every 4 batches (see how many times they accumulate for details). This means that when you set the batchsize to 16, the parameters will only be updated every 4 batches (see the accumulate I printed during training, the first time is the target value, the next time is the current value), i.e. the actual batchsize is the closest multiple of 64 to the batchsize you entered (here I modified the original code to strictly not exceed 64). So your input batchsize 17 (actual 51) is much smaller than 16 (actual 64), and you should take this into account when adjusting the learning rate. The current parameter is set to 18 on an 11G graphics card. Weakly modify the batchsize to observe the change of loss during warmup, and consider reducing the learning rate before accumulate to reach the target value of large oscillations.

common.py
This code is the common basic operation class in YOLOV5, in which I added ARM, FFM, RFB1, 2 of BiSeNet (non-RFBNet, see code comments for the magic modified version), ASPP (interface to increase the parameters used to cut the channel), ASPPs (first use 1 * 1 to reduce the input channel so that you can cut some intermediate channels less), Attention (channel attention, equivalent to ARM without 3×3 convolution, base SE), DAPPM (see HIT paper, the effect is not obvious here), PyramidPooling (PSPNet)

yolo.py
The main architecture code of the model for yolov5, including the Model class and the Detect class to be used for detection, I put the four newly added split header classes in this code (it might be more refreshing to put it in common.py). All newly added modules to be put into the model must go through the Model class, the following parts please focus on.
(1) Model's initialization function, I manually added 24 layers in save (split layer number, detection is 25). The original code forward_onece uses a for loop forward reasoning, the results of the subsequent layers will be used to save in the list (which layers will be used by parse function to parse the yaml configuration file to get, in the initialization function called parse, need to save the intermediate layer number in the save list, forward when the corresponding layer intermediate results in accordance with the save sequence number The results are stored in the y list), the current method, because I manually added 24 layers, the detection layer will return x (detection results) and y [-2] (segmentation results) after the end of the run. Therefore, if you modify the configuration file to add a new layer (for example, to the latest P6 model to add a segmentation layer), be sure to modify the initialization function of the Model to change 24 to the new segmentation layer number (this is not a good interface, rush, and do not change 24 to -2, see the original yolo code to know that this change does not work). In addition yolov5 original author in many code default detection layer is the last layer, must be in the configuration of the detection layer in the last layer.
(2) Model's parse_model parsing function from the yaml file parsing configuration, if you want to add a new module first in common.py or yolo.py in the implementation of the class, in parse_model in the parse method of writing the class, and then write the configuration in the configuration file. If you are designing a new split header after my split header class interface, just implement the class and add the class name to the list of support for parsing split headers in parse_model.

models/yolov5s_city_seg.yaml
model configuration file, you can see that I added the segmentation layer configuration in front of the detection layer and added the segmentation category (cityscapes is 19). Reasoning about different head pre-training models do not need to be modified, want to train different head models need to comment and uncomment (psp, base and lab do not need to change train.py again but bise also comment and uncomment train.py two places to add aux loss, the follow-up will explain the interface design flaws, but for the time being no time to change, in fact, with psp, base, lab is enough. s, m, l model reference to the original, the difference is only in the control of depth and width of depth_multiple, width_multiple values (base, psp and lab split head will also be automatically scaled down with s, m, l).

data/cityscapes_det.yaml
Test dataset configuration, same as the original version, with new segmented dataset address, train.py reads the segmented data address as configured here

test.py
New segmentation test function

utils/loss.py
New segmentation CE loss with aux (currently using this), segmentation Focal loss (more adequate experiments show that the effect is not good, at least 1 point lower), OHEM (theoretically should be better than CE, the actual lower than a few points, and the learning rate and loss ratio has a certain relationship, the gradient accumulation mechanism also seems to be a bit buggy), in short, most of the cases are recommended to use CE, the category is extremely unbalanced when then consider ohem and focal loss.

utils/metrics.py
Added fitness2 function for train when selecting models, including P, R, AP@.5, AP@.5:.95和mIoU的比例. Added mIoU function to calculate mIoU.

detect.py
Added functions for drawing segmentation and superposition maps, saving videos with the same size map, and converting trainid to id for submission (see inference section above), modified the case of opening cudnn.benchmark

SegmentationDataset.py
Segmentation data processing class. The original code is from the pytorch-ecoding project,, adding colorjittor, adding resize long-edge non-uniform sampling, modifying the crop method, modifying the way of testval mode, abolishing the val mode (much faster than testval mode, but the measured value is not accurate accuracy. The problem is that it is not very efficient to handle complicated loading, and the CPU and disk requirements are high (very slow on colab and kaggle). Training process may be stuck for a period of time or only test a child process, the program did not die, wait a short time on the good, belong to the normal phenomenon of bugs. Training other segmentation data such as BDD100k need to follow cityscapes inheritance base class (Cityscapes and bdd100k hybrid class has been implemented, as well as the example with custom_data class), especially the label conversion part, pay attention to the pad pixel for 255 and ordinary ignore category when loaded together with the conversion to -1, some Some dataset ids need to be converted to trainid (the current custom_data class is for data with no id conversion and ignore tag of 255, same as bdd100k).

train.py
The training process is to run a set of detection data backward for each batch, and then run a set of segmentation data backward, and then accumulate the unified update parameters. Every 10 rounds to measure the accuracy of the segmentation, the last 40 rounds of each round to measure the segmentation of the time to update best.pt. (The reason for this is because the loaders of testval mode is a bit of a problem leading to the death of some sub-processes, measuring the segmentation is very slow, my machine more than 1 minute). In addition, there is no support for multi-card training, temporarily can not use multiple cards.
The time relationship between ohem and CE interface is not consistent, the loop in the CE interface aux different number of inputs to find the loss is not consistent, replace the segmentation head training with aux loss when you want to annotate the annotation train.py these two places (marked with a long ----- comment).

