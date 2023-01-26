% This script generates a copy of the HSI-Drive 2.0 dataset using
% uint16 format for data codification

% Set MF to '1' for cubes with median filtering

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2022

MF = 0;
if MF
    npath = '..\Image_dataset\cubes_fl32\MF';
    path = 'Image_dataset\cubes_fl32\MF';
else
   npath = '..\Image_dataset\cubes_fl32';
   path = 'Image_dataset\cubes_fl32';
end
addpath(npath)

cd ..
cd(path)
files = dir('*.mat');
cd ..
numberOfFiles = length(files);
% Create folder for uint16 cubes
mkdir cubes_u16

if ~ isempty(files)
    for k=1:numberOfFiles
        cubef = load(files(k).name);
        cube = cubef.cube;
        clear cubef;
        cube_u16 = uint16(cube);
        dotposition = find(files(k).name == '.');
        noextname = files(k).name(1:dotposition(1)-1);
        filename = strcat('cubes_u16\',noextname,'_u16.mat');
        save(filename,"cube_u16");
    end
end
