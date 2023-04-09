function raw2jpg(filename, folder)

% Converts a raw image into a jpeg compressed image and saves into folder
% Example: raw2jpg('nf3123_153.bin', 'visible_raws')
%
% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath ..\Image_dataset\raw
fileID = fopen(filename);
A = fread(fileID,[2048 1088],'uint16')';
fclose(fileID);
AG = mat2gray(A);
[filepath,name,ext] = fileparts(filename);
imagename=strcat(name,'.jpg');
mkdir(folder);
imwrite(AG,fullfile(folder,imagename),'jpg');
%imwrite(AG,fullfile(folder,imagename),'jpg','BitDepth',12);
clear A AG
