# UCF101 Action Recognition
sample code for action recognition on UCF101

**UCF101.py** sampler supports autoaugment[1] when scarcity of frames in dataset.

## Requirements
*   torch>=1.6.0
*   torchvision>=0.7.0
*   tensorboard>=2.3.0

## Usage
download and extract frame from UCF101 videos. [UCF101 Frame Extractor](https://github.com/titania7777/UCF101FrameExtrcator)

train(resnet18)
```
python train.py --frames-path /path/to/frames --labels-path /path/to/labels --tensorboard-path /path/to/tensorboard --frame-size 224 --learning-rate 1e-3 --model resnet --uniform-frame-sample
```
train(r2plus1d18)
```
python train.py --frames-path /path/to/frames --tensorboard-path /path/to/tensorboard --model r2plus1d --uniform-frame-sample
```

## Settings and Results
**device information**: GTX Titan X (12GB)  

**Common option settings**  
list number: 1  
frame size: 112(r2plus1d), 224(resnet)  
num epochs: 30  
batch size: 16  
uniform frame sample: True  
random start position: False  
max interval: 7  
random interval: False  
sequence length: 30  
num layer: 1 (resnet)  
hidden size: 512 (resnet)  
learning rate: 5e-4(r2plus1d), 1e-3(resnet)  
scheduler step: 10  
scheduler gamma: 0.9  

**require video memory**: resnet: about 5642 MB, r2plus1d: about 4912 MB  

option | Accuracy
-- | -- 
resnet18 | 68.96
r2plus1d18  | 91.84

## ```UCF101.py``` Options
### common options
1. model: choose for different normalization value of model
2. frames_path: frames path
3. labels_path: labels path
4. frame_size: frame size(width and height are should be same)
5. sequence_length: number of frames
6. setname: sampling mode, if this mode is 'train' then the sampler read a 'train.csv' file to load train dataset [default: 'train', others: 'test']
### pad options
7. random_pad_sample: sampling frames from current frames with randomly for making some pads when frames are insufficient, if this value is False then only use first frame repeatedly [default: True, other: False]
8. pad_option: if this value is 'autoaugment' then pads will augmented by autoaugment policies [default: 'default', other: 'autoaugment']
### frame sampler options
9. uniform_frame_sample: sampling frames with same interval, if this value is False then sampling frames with ignored interval [default: True, other: False]
10. random_start_position: decides the starting point with randomly by considering the interval, if this value is False then starting point is always 0 [default: True, other, False]
11. max_interval: setting of maximum frame interval, if this value is high then probability of missing sequence of video is high [default: 7]
12. random_interval: decides the interval value with randomly, if this value is False then use a maximum interval [default: True, other: False]

## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  