PointLSTMX

## Prerequisites

These code is implemented in Pytorch (>1.0). Thus please install Pytorch first.
## Usage


### Data Preparation

#### SHREC'17

- Download the [SHREC'17 dataset](http://www-rech.telecom-lille.fr/shrec2017-hand/) [[Visualization]](https://github.com/Blueprintf/awesome-Gesture-Sign-Language-Recognition/blob/master/dataset/Overview_SHREC17.md) and put `HandGestureDataset_SHREC2017` directory to `./dataset/SHREC2017`, It is suggested to make a soft link toward downloaded dataset.
- Generate point cloud sequences from depth video, and save the processed point clouds in ```./dataset/Processed_SHREC2017```. Each video generate 32*256 points, and the generated point clouds occupy about 2.5G.
```bash
cd dataset
python shrec17_process.py
```
#### NvGesture

- Download the [NvGesture dataset](https://docs.google.com/forms/d/e/1FAIpQLSc7ZcohjasKVwKszhISAH7DHWi8ElounQd1oZwORkSFzrdKbg/viewform) [[visualization]](https://github.com/Blueprintf/awesome-Gesture-Sign-Language-Recognition/blob/master/dataset/Overview_NVGesture.md) and extract the NvGesture directory to `./dataset/Nvidia`, it is suggested to make a soft link toward downloaded dataset.
- Generate point cloud sequences from depth video, and save the processed point clouds in `./dataset/Nvidia/Processed`. Each video generate 32* 512 points, and the generated point clouds occupy about 11G.

```bash
cd dataset
python nvidia_dataset_split.py
python nvidia_process.py
```
#### MSRAction3D
The processed MSRAction3D dataset has been placed in the designated folder.

### Training


```python
cd experiments
python main.py --phase=train --work-dir=PATH_TO_SAVE_RESULTS --device=0 
```


### Inference

```python
cd experiments
python main.py --phase=test --work-dir=PATH_TO_SAVE_RESULTS --device=0 --weights=PATH_TO_WEIGHTS
```
### Citation

Please cite the following paper if you feel PointLSTMX useful to your research.



Relevant paper:An Efficient PointLSTM for Point Clouds Based Gesture Recognition
```latex
@inproceedings{min_CVPR2020_PointLSTM,
  title={An Efficient PointLSTM for Point Clouds Based Gesture Recognition},
  author={Min, Yuecong and Zhang, Yanxiao and Chai, Xiujuan and Chen, Xilin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5761--5770},
  year={2020}
}
```
