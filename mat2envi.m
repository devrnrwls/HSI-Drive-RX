function mat2envi(HSIcube, format, precision, sort_bands) 

% This function converts Matlab Level 5 .mat files containing HSI-Drive
% cubes to ENVI files
% HSIcube = name of the file containing the cube
% format = 'bil','bip' o 'bsq'
% precision = 'double', 'single' or 'uint16'
% sort_bands = '1' (yes) or '0' (no)

% example mat2envi('nf3123_153_MF_TC_N_u16.mat','bil','uint16',1)

% dependencies: 

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load cube
addpath ..\Image_dataset\cubes_uint16
addpath ..\Image_dataset\cubes_float32
addpath ..\Image_dataset\cubes
load(HSIcube);

if strcmp(precision,'uint16')
    data = permute(cube_u16 , [2 3 1]);
elseif strcmp(precision,'single')
	data = permute(cube_fl32 , [2 3 1]);
elseif strcmp(precision,'double')
    data = permute(cube , [2 3 1]);
else
    error('Error. \input format must be uint16, single or double')
end

%% sort bands?
bands = [888.48, 897.66, 879.13, 869.11, 956.11, 795.96, 807.63, 783.76,...
 770.58, 679.14, 744.48, 757.92, 731.88, 718.30, 692.90, 928.81, 936.25,...
 920.60, 912.17, 950.39, 848.55, 858.95, 838.36, 826.77, 944.49];
if sort_bands
    [sbands,ind]=sort(bands);
    data = data(:,:,ind);
end

%% Write ENVI file
dotposition = find(HSIcube == '.');
if isempty(dotposition)
            noextname = HSIcube;
        else
           noextname = HSIcube(1:dotposition(1)-1);
end
        
filename = sprintf('%s%s%s', noextname, '.', format);
%filename = sprintf('%s%s%s', noextname, '.dat');
    
multibandwrite(data,filename,format,'precision',precision,'machfmt','ieee-le');

%% Write ENVI header
headername = sprintf('%s%s%s', noextname, '.', 'hdr');
write_HSI_drive_hdr(headername, format, precision, sort_bands);

