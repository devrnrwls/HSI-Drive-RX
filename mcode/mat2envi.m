function mat2envi(HSIcube, format, precision, sort_bands, version) 

% This function converts Matlab Level 5 .mat files containing HSI-Drive
% cubes to ENVI files
% HSIcube = name of the file containing the cube
% format = 'bil','bip' o 'bsq'
% precision = 'single' or 'uint16'
% sort_bands = '1' (yes) or '0' (no)
% version = 'v20' or 'v21scale' or 'v21noscale'

% example mat2envi('nf3123_153_TC.mat','bil','uint16',1)
% NOTE: In the HSI-Drive 2.0 dataset all cubes are codified using 32-bit
% single precision. Use function single2u16 to generate a uint16 dataset.

% dependencies: 

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2025
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% load cube
if strcmp(version,'v20')
    addpath ..\Image_dataset\cubes_fl32
    addpath ..\Image_dataset\cubes_fl32\MF
elseif strcmp(version,'v21scale')
    addpath ..\Image_dataset\cubes_fl32\Cubes_Scaling
elseif strcmp(version,'v21noscale')
    addpath ..\Image_dataset\cubes_fl32\Cubes_NoScaling
    addpath ..\Image_dataset\cubes_fl32\Cubes_NoScaling\MF
end
load(HSIcube);

if strcmp(precision,'uint16')
    cube_u16 = uint16(cube);
    data = permute(cube_u16 , [2 3 1]);
elseif strcmp(precision,'single')
    data = permute(cube , [2 3 1]);
else
    error('Error. \input format must be uint16 or single')
end

%% sort bands? (values correspond to central frequencies of most
% contributing peaks)
bands = [888.48, 897.66, 879.13, 869.11, 956.11,...
        795.96, 807.63, 783.76, 770.58, 679.14,...
        744.48, 757.92, 731.88, 718.30, 692.90,...
        928.81, 936.25, 920.60, 912.17, 950.39,...
        848.55, 858.95, 564.36, 826.77, 944.49];
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
maxval = max(max(max(data)));
headername = sprintf('%s%s%s', noextname, '.', 'hdr');
write_HSI_drive_hdr(headername, format, precision, sort_bands, maxval);

