function GTdisplay(GT_file) 

% This function displays labeled images/ground-truth images
% Example: GTdisplay('nf3123_153_TC.png')
%
% Dependencies: HSIdrive_colormap.mat
%
% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath ..\Image_dataset\labels
pixel_labels = imread(GT_file);
load HSIdrive_colormap.mat;

figure(101)
fig_labels=figure(101);

GT=image(pixel_labels);
colormap(fig_labels,HSDcm);
caxis([0 11])
words = extractfield(GT_encoding,'surfaces');
colorbar('Location','eastoutside','Ticks',[0.5,1.5,2.5,3.5,4.5,5.5,6.5,...
    7.5,8.5,9.5,10.5],'TickLabels',words)
set(gca,'DataAspectRatio',[1 1 1])
set(gca,'LooseInset',get(gca,'TightInset'))
title(strcat(GT_file,': Ground-truth image'));
