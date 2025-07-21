# ICCV2025-GDL

Here is the official code for "Boosting Multimodal Learning via Disentangled Gradient Learning", which is a flexible framework to enhance the optimization process of multimodal learning. Please refer to our [ICCV 2025 paper](https://arxiv.org/pdf/2507.10213) for more details.


## Main Dependencies
+ Ubuntu 20.04
+ CUDA Version: 11.1
+ PyTorch 1.11
+ python 3.8.6
+ **note** The optimal hyperparameter ($\alpha$) for DGL may vary with the dependencies. If your equipment and software are different, you may need to adjust the hyperparameters accordingly.


## Usage
### Data Preparation
Download Dataset：
[CREMA-D](https://pan.baidu.com/s/11ISqU53QK7MY3E8P2qXEyw?pwd=4isj), [Kinetics-Sounds](https://pan.baidu.com/s/1E9E7h1s5NfPYFXLa1INUJQ?pwd=rcts).
Here we provide the processed dataset directly. 

The original dataset can be seen in the following links,
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset).
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/),

 And you need to process the dataset following the instruction below.

### Pre-processing

For CREMA-D and VGGSound dataset, we provide code to pre-process videos into RGB frames and audio wav files in the directory ```data/```.

#### CREMA-D 

As the original CREMA-D dataset has provided the original audio and video files, we simply extract the video frames by running the code:

```python data/CREMAD/video_preprecessing.py```

Note that, the relevant path/dir should be changed according your own env.  

#### VGGSound

As the original VGGSound dataset only provide the raw video files, we have to extract the audio by running the code:

```python data/VGGSound/mp4_to_wav.py```

Then, extracting the video frames:

```python data/VGGSound/video_preprecessing.py```

Note that, the relevant path/dir should be changed according your own env. 

## Data path

you should move the download dataset into the folder *train_test_data*, or make a soft link in this folder.


## Key point of DGL

- **Unimodal Regularization**. Add unimodal loss to enhance the modality encoder via  the parameters-shared method. （see **_AUXI in models/fusion_modules.py）
- **Multimodal Gradient Truncation**. Remove the gradient from the multimodal loss to the modality encoder （see **.detach()** in fusion module in the models/fusion_modules.py）
- **Unimodal Gradient Truncation**. Remove the gradient from the unimodal loss to the multimodal fusion module (see Line 114 in main_dgl.py)

## Train 

We provide bash file for a quick start.

For CREMA-D

```bash
bash cramed_dgl.sh
```


## Test

```python
python valid.py
```

## Contact us

If you have any detailed questions or suggestions, you can email us:
**shicaiwei@std.uestc.edu.cn**
