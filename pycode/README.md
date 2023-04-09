# Training and testing

## Requirements

The packages needed to run these scripts can be found in the 2.5 version of the Vitis AI containers provided by AMD-Xilinx.
See [github.com/Xilinx/Vitis-AI/tree/2.5](https://github.com/Xilinx/Vitis-AI/tree/2.5).

There is a prebuilt [CPU](https://hub.docker.com/layers/xilinx/vitis-ai-cpu/2.5/images/sha256-eaa85efb06924995ebdb973546e7f69169b003b8cc525764bd9524ad554dddbe?context=explore) image,
but a [GPU](https://github.com/Xilinx/Vitis-AI/tree/2.5#building-docker-from-recipe) one can also be built.

Once you run the container, you have to choose `vitis-ai-tensorflow2` environment.

Alternatively, you can create your own Conda environment.

## Working directory

```text
# Your working directory
    ├── dataset
        ├── Train
            ├── PatchesNpy
            ├── LabelsExp1
            ├── LabelsExp2
        ├── Val
            ├── PatchesNpy
            ├── LabelsExp1
            ├── LabelsExp2
        ├── Test
            ├── PatchesNpy
            ├── LabelsExp1
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

The pretrained models for the 3-class and 5-class experiments can be found under `/training/pretrained-models` directory.

TC_PN: In the cube generation process apart from clipping and cropping the raw image and performing reflectance
correction steps, traslation to center (TC) and pixel normalization (PN) have also been applied.

2_3_8: This three values are related to the depth of the U-Net, the convolution kernel size and the number of filter of
the first convolution block.

## Dataset downloading

The dataset (HSI-Drive v1.1) that has been used to train and test the pretrained models can be found in
[HSI-Drive](https://ipaccess.ehu.eus/HSI-Drive/#download), as well as the recently uploaded v2.0.

NOTE: The scripts that can be found in this repository expect the data to be in `.npy` format, while the downloaded
cubes have a `.m` format.

## Testing

When testing, apart from reporting the metrics, the resulting output images are saved in a folder specified by the user.

```
python3 test.py -fm 'pretrained_models/float_model_3classes_TC_PN_2_3_8_explicit_norm_4.h5' -pd '../dataset/Pred/' -ex 1
```

## Training from scratch

To train this FCN from scratch just run

```
python3 train.py -fm './model.h5' -ex 2
```

## Citing

This pretrained models have been used in [Exploring Fully Convolutional Networks for the Segmentation of Hyperspectral Imaging Applied to Advanced Driver Assistance Systems](https://link.springer.com/book/10.1007/978-3-031-12748-9).
If you find this research useful, cite it as:

```
@inproceedings{gutierrez2022exploring,
  title={Exploring Fully Convolutional Networks for the Segmentation of Hyperspectral Imaging Applied to Advanced Driver Assistance Systems},
  author={Guti{\'e}rrez-Zaballa, Jon and Basterretxea, Koldo and Echanobe, Javier and Mart{\'\i}nez, M Victoria and del Campo, In{\'e}s},
  booktitle={Design and Architecture for Signal and Image Processing: 15th International Workshop, DASIP 2022, Budapest, Hungary, June 20--22, 2022, Proceedings},
  pages={136--148},
  year={2022},
  organization={Springer}
}
```
