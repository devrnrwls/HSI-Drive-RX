function envi2mat(envifile,precision)

% Convert HDI-drive ENVI file to Matlab Level 5 .mat files containing
% HSI-drive cubes
% envifile = name of the file containing the cube ('nfECHV_processing.bil')
% format = 'bil','bip' o 'bsq'
% precision = 'double', 'single' or 'uint16'

% example envi2mat('nf3123_153_MF_TC_N_u16.bil','uint16');

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dotposition = find(envifile == '.');
if isempty(dotposition)
    error('Error. \no format extension')
else
    format =  sprintf('%s',envifile(dotposition+1:dotposition+3));
    noextname = envifile(1:dotposition-1);
end

filename = sprintf('%s%s', noextname, '.mat');

if strcmp(precision,'uint16')
    cube_u16 = multibandread(envifile,[216 409 25],precision,0,format,'ieee-le');
    cube_u16 = permute(cube_u16,[3 1 2]);
    save(filename,'cube_u16');
elseif strcmp(precision,'single')
    cube_fl32 = multibandread(envifile,[216 409 25],precision,0,format,'ieee-le');
    cube_fl32 = permute(cube_fl32,[3 1 2]);
    save(filename,'cube_fl32');
elseif strcmp(precision,'double')
    cube = multibandread(envifile,[216 409 25],precision,0,format,'ieee-le');
    cube = permute(cube_u16,[3 1 2]);
    save(filename,'cube');
else
    error('specified numeric format must be double, single or uint16');
end
