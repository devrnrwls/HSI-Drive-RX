# EHU-GDED/FLOSS/HSI-Drive

<p align="center">
  <a title="WebSite" href="https://ipaccess.ehu.eus/HSI-Drive"><img src="https://img.shields.io/website.svg?label=ipaccess.ehu.eus%2FHSI-Drive&longCache=true&style=flat-square&url=https%3A%2F%2Fipaccess.ehu.eus%2FHSI-Drive%2Findex.html"></a>
</p>

- `mcode/`:
  Matlab functions and scripts to process v2 the dataset.

- `pycode/`:
  Python functions and scripts for training and testing a modified version of [U-Net](https://en.wikipedia.org/wiki/U-Net)
  to segment images from driving scenarios.

  Pretrained floating-point models to segment 3 classes (Road, Road Marks and No Drivable) and 5 classes (Road, Road Marks,
  Vegetation, Sky and Others) which have been trained using v1.0 of HSI-Drive dataset and one pretrained floating-point
  model to segment the same 5 classes but trained with v2.0 of the dataset are available at
  [ipaccess.ehu.eus/HSI-Drive](https://ipaccess.ehu.eus/HSI-Drive).
  See [pycode/testing/get_pretrained_models.sh](pycode/testing/get_pretrained_models.sh).
