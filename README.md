# EHU-GDED/FLOSS/HSI-Drive

Open source scripts for processing the HSI-Drive dataset.
See [ipaccess.ehu.eus/HSI-Drive](https://ipaccess.ehu.eus/HSI-Drive/).

- `/mcode/`:
  Matlab functions and scripts to process the data in the *HSI-Drive v2.0* dataset.

- `/pycode/`:
  Python functions and scripts for training and testing a modified version of [U-Net](https://en.wikipedia.org/wiki/U-Net)
  to segment images from driving scenarios.

  Pretrained floating-point models to segment 3 classes (Road, Road Marks and No Drivable) and 5 classes (Road, Road Marks,
  Vegetation, Sky and Others) which have been trained using v1.0 of HSI-Drive dataset and one pretrained floating-point
  model to segment the same 5 classes but trained with v2.0 of the dataset are available at
  [ipaccess.ehu.eus/HSI-Drive](https://ipaccess.ehu.eus/HSI-Drive).
  See [pycode/testing/get_pretrained_models.sh](pycode/testing/get_pretrained_models.sh).
