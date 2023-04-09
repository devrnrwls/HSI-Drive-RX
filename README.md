# HSI-Drive Matlab functions

These functions help processing the data in the HSI-Drive dataset.

- `denormalize.m`:
  Denormalize the cubes that were generated with final band normaliztion and saves the denormalized cube in a new file.
  Coeffiecient `.mat` files for denormalization of each image are in the "Coefficients" folder.

- `envi2mat.m`:
  Convert ENVI files to Matlab Level5 `.mat` matrices containing the hyperspetral information of an image.

- `GTdisplay.m`:
  Display the selected ground-truth image using HSI-Drive labeling color-code.

  - `HSIdrive_colormap.mat`

- `mat2envi.m`:
  Convert the Matlab Level5 `.mat` matrices containing the hyperspetral information of an image to ENVI format binary files
  (including the generation of ENVI headers).

  - `write_HSI_drive_hdr.m`:
    Generate ENVI headers.

- `rawsequence2avi.m`:
  Convert provided example raw image sequences into AVI videos for easy visualization.

  - `raw2jpg.m`:
    Convert original camera raw files into grayscale images and saves them using jpeg encoding.

- `select_HSI_files.m`:
  Generate subsets of the HSI-Drive image dataset and their corresponding ground-truths, according to the four parameters
  used in the dataset organization: Season, Weather, Daytime and Scene.
  The generated subset is saved to a new folder.
