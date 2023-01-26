# HSI-Drive Matlab functions

These functions help processing the data in the HSI-Drive dataset.

- `GTdisplay.m`: This function displays the selected ground-truth image using HSI-Drive labeling color-code.
  - `HSIdrive_colormap.mat`

- `raw2jpg.m`: This function converts original camera raw files into grayscale images and saves them using jpeg encoding.

- `select_HSI_files.m`: This function generates subsets of the HIS-Drive image dataset according to the four parameters used in the dataset organization: Season, Weather, Daytime and Scene. The generated subset is saved to a new folder.

- `mat2envi.m`: This function converts the Matlab Level5 .mat matrices containing the hyperspetral information of an image to ENVI format binary files (including the generation of ENVI headers).
  - `write_HSI_drive_hdr.m`

- `envi2mat.m`: This function converts ENVI files to Matlab Level5 .mat matrices containing the hyperspetral information of an image.

- `rawsequence2avi.m`: This script converts provided example raw image sequences into AVI videos for easy visualization.
