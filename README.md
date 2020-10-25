# UCF101 Action Recognition
sample code for action recognition on UCF101

```UCF101.py``` sampler supports autoaugment[1] when scarcity of frames in dataset.

## Requirements
*   torch>=1.6.0
*   torchvision>=0.7.0
*   tensorboard>=2.3.0

## Usage
download and extract frame from UCF101 videos. [UCF101 Frame Extractor](https://github.com/titania7777/UCF101FrameExtrcator)

train(resnet18)
```
python train.py --frames-path /path/to/frames --tensorboard-path /path/to/tensorboard --model resnet --random-pad-sample --uniform-frame-sample --random-start-position --random-interval --bidirectional
```
train(r2plus1d18)
```
python train.py --frames-path /path/to/frames --tensorboard-path /path/to/tensorboard --model r2plus1d --random-pad-sample --uniform-frame-sample --random-start-position --random-interval
```
## Settings and Results
```device information```  
GPU: GTX Titan X (12GB)   

```Common option settings```  
list number: 1  
frame size: 224  
num epochs: 20  
batch size: 16  
uniform frame sample: True  
random start position: True  
max interval: 7  
random interval: True  
sequence length: 15  
num layer: 1 (for lstm)  
hidden size: 512 (for lstm)  
bidirectional: True (for lstm)  
learning rate: 5e-4 (SGD)  
scheduler step: 10 (per epoch)  
scheduler gamma: 0.5  

```require video memory```  
resnet: about 11677 MB  
r2plus1d: about 10042 MB  

option | Accuracy
-- | -- 
resnet18(pad-option: default) | None
resnet18(pad-option: autoaugment) | None
r2plus1d18(pad-option: default)  | None
r2plus1d18(pad-option: autoaugment)  | None

## ```UCF101.py``` Options
### common options
1. model: choose for different normalization value of model
2. frames_path: frames path
3. labels_path: labels path
4. frame_size: frame size(width and height are should be same)
5. sequence_length: number of frames
6. setname: sampling mode, if this mode is 'train' then the sampler read a 'train.csv' file to load train dataset [default: 'train', others: 'test']
### pad options
7. random_pad_sample: sampling frames from existing frames with randomly when frames are insufficient, if this value is False then only use first frame repeatedly [default: True, other: False]
8. pad_option: when adds some pad for insufficient frames of video, if this value is 'autoaugment' then pads will augmented by autoaugment policies [default: 'default', other: 'autoaugment']
### frame sampler options
9. uniform_frame_sample: sampling frames with same interval, if this value is False then sampling frames with ignored interval [default: True, other: False]
10. random_start_position: decides the starting point with randomly by considering the interval, if this value is False then starting point is always 0 [default: True, other, False]
11. max_interval: setting of maximum frame interval, if this value is high then probability of missing sequence of video is high [default: 7]
12. random_interval: decides the interval value with randomly, if this value is False then use a maximum interval [default: True, other: False]

## references
-------------
[1] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, "AutoAugment: Learning Augmentation Strategies From Data", Computer Vision and Pattern Recognition(CVPR), 2019, pp. 113-123  