function denormalize(cubenorm, coefs)

% This function denormalizes the cubes thet were generated with final band
% normaliztion and save the denormalized cube in a file
% Inputs:
%   cubenorm: file containing the normalized cube
%   coefs: normalization coefficients (mapminmax)

% Example: denormalize('nf4223_283_MF_TC_N.mat','nf4223_283_MF_TC_N_Coef')

% Author: Koldo Basterretxea
% GDED group
% University of the Basque Country (UPV/EHU)
% January 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load cube with band normaliztion (cube)
load(cubenorm)
% Load normaliztion coeficients (NCoef)
load(coefs)
% Convert cube into 2D matrix
[b,r,c]=size(cube);
cube2D = reshape(cube,[b,r*c]);
% Denormalize bands
nonorm = mapminmax('reverse',cube2D,NCoef);
% Generate new denormalized cube
cube = reshape(nonorm,[b,r,c]);

%Save denormalized cube
dotpos = find(cubenorm == '.');
filename = sprintf('%s',cubenorm(1:dotpos-3));
save(filename,'cube');