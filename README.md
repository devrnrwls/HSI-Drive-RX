# EHU-GDED/FLOSS/HSI-Drive

Open source scripts for processing the HSI-Drive dataset.
See [ipaccess.ehu.eus/HSI-Drive](https://ipaccess.ehu.eus/HSI-Drive/).

- `/mcode/`:
  Matlab functions and scripts to process the data in the *HSI-Drive v2.0* dataset.

- `/pycode/`:
  Python functions and scripts for training and testing a modified version of [U-Net](https://en.wikipedia.org/wiki/U-Net)
  to segment images from driving scenarios.
  It also contains two pretrained floating-point models to segment 3 classes (Road, Road Marks and No Drivable) and 5 classes (Road, Road Marks, Vegetation, Sky and Others).
