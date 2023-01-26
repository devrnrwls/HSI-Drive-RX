function select_HSI_files(season, weather, daytime, scene, precision)

% This function selects a subset of the HSI-Drive dataset files according
% to specified annotation fields and saves them to a new folder

% SEASON: 'spring','summer','fall','winter','all'
% WEATHER: 'sunny', 'cloudy', 'wet','foggy','all'
% DAYTIME: 'dawn','midday','sunset','all'
% SCENE: 'road', 'highway','urban','all'
% precision: 'fl32' for single, 'u16' for uint16
%% NOTE: In the HSI-Drive 2.0 dataset all cubes are codified using 32-bit
% single precision. Use function single2u16 to generate a uint16 dataset.

% Example: select_HSI_files('spring', 'all', 'all', 'road','float32')

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2022

%% Use HSI-Drive file annotation code
switch season
    case 'spring'
        se='3';
    case 'summer'
        se='4';
    case 'fall'
        se='2';
    case 'winter'
        se='1';
    case 'all'
        se='?';
    otherwise
        error('type correct season')
end
switch weather
    case 'sunny'
        we='1';
    case 'cloudy'
        we='2';
    case 'wet'
        we='3';
    case 'foggy'
        we='4';
    case 'all'
        we='?';
    otherwise
        error('type correct weather')
end
switch daytime
    case 'dawn'
        dt='1';
    case 'midday'
        dt='2';
    case 'sunset'
        dt='3';
    case 'all'
        dt='?';
    otherwise
        error('type correct daytime')
end
switch scene
    case 'urban'
        sc='1';
    case 'road'
        sc='2';
    case 'highway'
        sc='3';
    case 'all'
        sc='?';
    otherwise
        error('type correct scene')
end

%% Files to be copied
filenames = strcat('**\nf',se,we,dt,sc,'_*');
cd ..
folder = strcat('Image_dataset\cubes_',precision);
%folder_GTs = 'Image_dataset\labels';
cd(folder)
files = dir(filenames);
cd ..
cd labels
GTs = dir(filenames);
numberOfFiles = length(files);
cd ..

% Create folder for file subset
foldername = strcat(season,'_',weather,'_',daytime,'_',scene);
mkdir(foldername)

% Copy files to new folders
subfolderold='none';
if ~ isempty(files)
    k2=0;
    for k=1:numberOfFiles
        k2 = k2+1;
        separator=find(files(k).folder == '\');
        subfolder = files(k).folder(separator(length(separator)):length(files(k).folder));
        subfoldername=strcat(foldername,subfolder);
        if ~strcmp(subfolderold,subfolder)
           mkdir(subfoldername)
        end
        subfolderold = subfolder;
        copyfile(strcat(files(k).folder,'\',files(k).name),subfoldername)
        % One GT file for two cubes (with MF and with no MF)
        if k == (numberOfFiles/2)+1
            k2 = 1;
        end
        copyfile(strcat(GTs(k2).folder,'\',GTs(k2).name),subfoldername)
    end
end

