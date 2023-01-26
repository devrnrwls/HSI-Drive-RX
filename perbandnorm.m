function [cubenorm, Ncoef] = perbandnorm(cubefile,minval,maxval)

% This function performs per-band normalization of HSI cubes
% cubefile: name of the .mat file containing 3D cube data
% minval: minimum value for band normalization
% maxval: maximum value for band normalization
%
% Example: [cubenorm, coefs] = perbandnorm('nf3123_153_TC.mat',0,(2^12)-1)
%
%
% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cd ..
addpath ../Image_dataset/cubes_fl32
% cd Matlab_functions

load(cubefile);
[sp,r,c]=size(cube);
cube2D = reshape(cube,[sp,r*c]);
% Normalize bands (rows)
[cube2Dn, Ncoef] = mapminmax(cube2D, minval, maxval);
% Generate new normalized cube
cubenorm = reshape(cube2Dn,[sp,r,c]);
% Save to mat file
dotposition = find(cubefile == '.');
noextname = cubefile(1:dotposition(1)-1);
normalizedfile = strcat(noextname,'_N.mat');
save(normalizedfile,'cubenorm');
