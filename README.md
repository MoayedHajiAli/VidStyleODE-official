# VidStyleODE Disentangled Video Editing via StyleGAN and NeuralODEs (ICCV 2023)

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://cyberiada.github.io/VidStyleODE/)
[![arXiv](https://img.shields.io/badge/Arxiv-2304.06020-b31b1b)](https://arxiv.org/abs/2304.06020)


<img src="assets/teaser.jpg" width="1000"/>

### VidStyleODE Disentangled Video Editing via StyleGAN and NeuralODEs (ICCV 2023)

<div align="justify">
<b>Abstract</b>: We propose VidStyleODE, a spatiotemporally continuous disentangled Video representation based upon StyleGAN and Neural-ODEs. Effective traversal of the latent space learned by Generative Adversarial Networks (GANs) has been the basis for recent breakthroughs in image editing. However, the applicability of such advancements to the video domain has been hindered by the difficulty of representing and controlling videos in the latent space of GANs. In particular, videos are composed of content (i.e., appearance) and complex motion components that require a special mechanism to disentangle and control. To achieve this, VidStyleODE encodes the video content in a pre-trained StyleGAN W+ space and benefits from a latent ODE component to summarize the spatiotemporal dynamics of the input video. Our novel continuous video generation process then combines the two to generate high-quality and temporally consistent videos with varying frame rates. We show that our proposed method enables a variety of applications on real videos: text-guided appearance manipulation, motion manipulation, image animation, and video interpolation and extrapolation. For more details, please visit our <a href='https://cyberiada.github.io/VidStyleODE/'>project webpage</a> or read our 
<a href='https://arxiv.org/abs/2304.06020'>paper</a>.
</div> 
<br>

## Content
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
    - [Dataset Download](#downloading-and-arranging-training-datasets)
    - [Setup StyleGAN Generator](#setup-stylegan2-generator)
    - [Setup StyleGAN2 Inversion](#setup-stylegan2-inversion)
    - [Training and Validation Splits](#training-validation-split)
3. [Training](#training)
4. [Applications](#applications)
    - [Image Annimation](#image-animation)
    - [Apperance Manipulation](#appearance-manipulation)
    - [Frame Interpolation](#frame-interpolation)
    - [Frame Extrapolation](#frame-extrapolation)
8. [Citation](#citation)

# Environment Setup
- initialize and activate a new conda environment by running 
```
conda create -n vidstyleode python=3.10
conda activate vidstyleode
```
- Install the requirements by running 
```
pip install -r requirements.txt
```

# Dataset Preparation 

### Downloading and Arranging Training Datasets
Please refer to [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)and [Fashion Dataset](https://github.com/leventkaracan/dicomogan) official websites for instructions on downloading the datasets used in the paper. You may also experiment with your own dataset. 
The datasets should be arranged with the following structure
```shell
Folder1
    Video_1.mp4
    Video_2.mp4
    ..
Folder2
    Video_1.mp4
    Video_2.mp4
    ..
```
It is recommended to extract the frames of the video for easier training. To extract the frames, please run the following command 
```shell
python scripts/extract_video_frames.py \
     --source_directory <path-to-video-directory> \
     --target_directory <path-to-output-target-directory>
```
The output folder will have the following structure 
```shell
Folder1_1
    000.png
    001.png
    ..
Folder1_2
    000.png
    001.png
    ..
```

### Setup StyleGAN2 Generator
- Our method relies on a pretrained StyleGAN2 generation. Please download your pretrained generator checkpoint and provide its path in the training configuration file. 
- For Face video (RAVDESS), we relied on the rosinality pretrained checkpoint. A converted checkpoint can be accessed from the StyleCLIP official repository, which can be downloaded from [here](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing).
- For full-body videos (Fashion Dataset), we relied on the pretrained checkpoint provided by [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human).


### Setup StyleGAN2 Inversion
- For memory efficiency and to reduce the computation during training, we precomput the StyleGAN W+ embedding vector. 
- **Frames Proprocessing**: - It is important to center-align your video frames before applying inversion. This is because stylegan generators usually generate aligned frames. Image inversion piplelines typically center-align the images before applying the stylegan inversion. If your videos are not center-aligned, **please replace your video frames with those aligned.**
- We rely on the official checkpoint of the [pSp Inversion](https://github.com/eladrich/pixel2style2pixel) for our experiments on face videos (RAVDESS), and on the official checkpoint from [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human) for our experiments on full-body videos (Fashion Dataset).
- Please refer to their official repositories for instructions on extracting the StyleGAN2 W+ embeddings. An embedding vector is typically of the shape `1 x 18 x hidden_dims`
- The embeddings should be saved as `.pt` files and arranged in a structure similar to the video frames. 
```
Folder1_1
 000.pt
 001.pt
 ..
Folder1_2
 000.pt
 001.pt
 ..
```


### (Optional) Setup Textual Descriptions
To enable style editing, you need to provide a textual description for each training video. Please store these descriptions in a file named `text_descriptions.txt` within the corresponding video frames folder. For example:
```
Folder1_1
 000.pt
 001.pt
 ..
 text_descriptions.txt
```

### Training Validation Split
- Prepare a `.txt` file containing the video folder names for the training and validation.
- Our splits for RAVDESS and Fasion Dataset are provided under the [data](./data/) folder. 

# Training
- Prepare a `.yaml` configuration file where you need to specify the video frames directory under `img_root`, the W+ inversion folder under `inversion_root`, and the training and validation `txt` files under `video_list`. 
- Our config files for the RAVDESS and Fashion Dataset are provided under the [configs](./configs/) folder.
- To start the training, run the following command:
```
python main.py --name <tag-for-your-experiment> \
               --base <path-to-config-file>
```

By default, the training checkpoint and figures will be logged under `logs` folder as well as into wandb. Therefore, please log in to wandb by running 
```
wandb login
```

# Applications 
## Image Animation
To generate image animation results by using the motion from a driving video, please run the following script 

```shell
python scripts/image_animation.py
    --model_dir <log-dir-to-pretrained-model> \
 --n_samples <number-of-sample-to-generate> \
    --output_dir <path-to-save-dir> \
 --n_frames <num-of-frames-to-generate-per-video> \
    --spv <num-of-dirving-videos-per-sample> \ # driving videos will be chosen randomly
    --video_list <txt-file-of-possible-target-videios> \
 --img_root <path-to-videos-root-dir> \
    --inversion_root <path-to-frames-inversion-root-dir> \
```
## Appearance Manipulation
Instructions will be added later.

## Frame Interpolation
Instructions will be added later.

## Frame Extrapolation
Instructions will be added later.


# Citation
If you find this paper useful in your research, please consider citing:
```
@misc{ali2023vidstyleodedisentangledvideoediting,
 title={VidStyleODE: Disentangled Video Editing via StyleGAN and NeuralODEs}, 
 author={Moayed Haji Ali and Andrew Bond and Tolga Birdal and Duygu Ceylan and Levent Karacan and Erkut Erdem and Aykut Erdem},
 year={2023},
 eprint={2304.06020},
 archivePrefix={arXiv},
 primaryClass={cs.CV},
 url={https://arxiv.org/abs/2304.06020}, 
}
```