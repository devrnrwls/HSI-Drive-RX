# Training and testing

## Requirements

The packages needed to run these scripts can be found in the Vitis AI containers provided by AMD-Xilinx in [v3.5](https://github.com/Xilinx/Vitis-AI/tree/v3.5).

There are some prebuilt [CPU docker images](https://hub.docker.com/layers/xilinx/vitis-ai-cpu/2.5/images/sha256-eaa85efb06924995ebdb973546e7f69169b003b8cc525764bd9524ad554dddbe?context=explore) for previous Vitis-AI versions which should be also compatible. All GPU images have to be built [from scratch](https://github.com/Xilinx/Vitis-AI/tree/v3.5/docker).

Once you run the container, you have to choose `vitis-ai-tensorflow2` environment.

Alternatively, you can create your own Conda environment.

## Working directory

```text
# Your working directory
    ├── dataset
        ├── Train
            ├── Cube_TC_PN_Npy
            ├── LabelsExp2
        ├── Val
            ├── Cube_TC_PN_Npy
            ├── LabelsExp2
        ├── Test
            ├── Cube_TC_PN_Npy
            ├── LabelsExp2
        ├── Pred
    ├── training
        ├── train.py
    ├── testing
        ├── test.py
    ├── processing
        ├── *.m
    ├── your_docker_run.sh
    ├── *.json
    ├── *.py
    ├── README.md

```

## Pretrained-models

The pretrained models for the 5-class experiments can be found under `/training/pretrained-models/v2.0` directory.

TC_PN: In the cube generation process apart from clipping and cropping the raw image and performing reflectance
correction steps, traslation to center (TC) and pixel normalization (PN) have also been applied.

4_3_32: This three values are related to the depth of the U-Net, the convolution kernel size and the number of filter of
the first convolution block.

## Dataset downloading

The dataset (HSI-Drive v2.0) that has been used to train and test the pretrained models can be found in
[HSI-Drive](https://ipaccess.ehu.eus/HSI-Drive/#download).

## Testing

When testing, apart from reporting the metrics, the resulting output images are saved in a folder specified by the user.

```
python3 test.py -fm 'pretrained_models/v2.0/float_model_5classes_TC_PN_4_3_32_explicit_norm_2.h5' -pd '../dataset/Pred/' -ex 2
```

## Training from scratch

To train this FCN from scratch just run

```
python3 train.py -fm './model.h5' -ex 2
```

## Citing

This pretrained models have been used in [Rapid Deployment of Domain-specific Hyperspectral Image Processors with Application to Autonomous Driving](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10382745).
If you find this research useful, cite it as:

```
@article{GUTIERREZZABALLA2023102878,
title = {On-chip hyperspectral image segmentation with fully convolutional networks for scene understanding in autonomous driving},
journal = {Journal of Systems Architecture},
volume = {139},
pages = {102878},
year = {2023},
issn = {1383-7621},
doi = {https://doi.org/10.1016/j.sysarc.2023.102878},
url = {https://www.sciencedirect.com/science/article/pii/S1383762123000577},
author = {Jon Gutiérrez-Zaballa and Koldo Basterretxea and Javier Echanobe and M. Victoria Martínez and Unai Martinez-Corral and Óscar Mata-Carballeira and Inés {del Campo}},
keywords = {Hyperspectral imaging, scene understanding, fully convolutional networks, autonomous driving systems, system on chip, benchmarks},
abstract = {Most of current computer vision-based advanced driver assistance systems (ADAS) perform detection and tracking of objects quite successfully under regular conditions. However, under adverse weather and changing lighting conditions, and in complex situations with many overlapping objects, these systems are not completely reliable. The spectral reflectance of the different objects in a driving scene beyond the visible spectrum can offer additional information to increase the reliability of these systems, especially under challenging driving conditions. Furthermore, this information may be significant enough to develop vision systems that allow for a better understanding and interpretation of the whole driving scene. In this work we explore the use of snapshot, video-rate hyperspectral imaging (HSI) cameras in ADAS on the assumption that the near infrared (NIR) spectral reflectance of different materials can help to better segment the objects in real driving scenarios. To do this, we have used the HSI-Drive 1.1 dataset to perform various experiments on spectral classification algorithms. However, the information retrieval of hyperspectral recordings in natural outdoor scenarios is challenging, mainly because of deficient color constancy and other inherent shortcomings of current snapshot HSI technology, which poses some limitations to the development of pure spectral classifiers. In consequence, in this work we analyze to what extent the spatial features codified by standard, tiny fully convolutional network (FCN) models can improve the performance of HSI segmentation systems for ADAS applications. In order to be realistic from an engineering viewpoint, this research is focused on the development of a feasible HSI segmentation system for ADAS, which implies considering implementation constraints and latency specifications throughout the algorithmic development process. For this reason, it is of particular importance to include the study of the raw image preprocessing stage into the data processing pipeline. Accordingly, this paper describes the development and deployment of a complete machine learning-based HSI segmentation system for ADAS, including the characterization of its performance on different embedded computing platforms, including a single board computer, an embedded GPU SoC and a programmable system on chip (PSoC) with embedded FPGA. We verify the superiority of the FPGA-PSoC over the GPU-SoC in terms of energy consumption and, particularly, processing latency, and demonstrate that it is feasible to achieve segmentation speeds within the range of ADAS industry specifications using standard development tools.}}

@INPROCEEDINGS{10382745,
  author={Gutiérrez-Zaballa, Jon and Basterretxea, Koldo and Echanobe, Javier and Mata-Carballeira, Óscar and Martínez, M. Victoria},
  booktitle={2023 30th IEEE International Conference on Electronics, Circuits and Systems (ICECS)},
  title={Rapid Deployment of Domain-specific Hyperspectral Image Processors with Application to Autonomous Driving*},
  year={2023},
  volume={},
  number={},
  pages={1-6},
  keywords={Road transportation;Quantization (signal);Costs;Program processors;Power demand;Image color analysis;Clouds;hyperspectral image processor;custom quantization;fully convolutional networks;autonomous driving},
  doi={10.1109/ICECS58634.2023.10382745}}

@INPROCEEDINGS{10371793,
  author={Gutiérrez-Zaballa, Jon and Basterretxea, Koldo and Echanobe, Javier and Victoria Martínez, M. and Martinez-Corral, Unai},
  booktitle={2023 IEEE Symposium Series on Computational Intelligence (SSCI)},
  title={HSI-Drive v2.0: More Data for New Challenges in Scene Understanding for Autonomous Driving},
  year={2023},
  volume={},
  number={},
  pages={207-214},
  keywords={Image segmentation;Technological innovation;Pedestrians;Computational modeling;Video sequences;Throughput;Robustness;hyperspectral imaging;dataset;scene understanding;autonomous driving systems;fully convolutional networks},
  doi={10.1109/SSCI52147.2023.10371793}}

```
