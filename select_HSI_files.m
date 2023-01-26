function select_HSI_files(season, daytime, weather, scene)

% This function selects a subset of the HSI-Drive dataset files according
% to specified annotation fields and saves them to a new folder

% SEASON: 'spring','summer','fall','winter','all'
% WEATHER: 'sunny', 'cloudy', 'wet','foggy','all'
% DAYTIME: 'dawn','midday','sunset','all'
% SCENE: 'road', 'highway','urban','all'
% Example: select_HSI_files('spring', 'all', 'all', 'road')

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2021

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
cd Image_dataset
files= dir(filenames);
numberOfFiles = length(files);

% Create folder for file subset
foldername = strcat(season,'_',weather,'_',daytime,'_',scene);
mkdir(foldername)

% Copy files to new folders
subfolderold='none';
if ~ isempty(files)
    for k=1:numberOfFiles
        separator=find(files(k).folder == '\');
        subfolder = files(k).folder(separator(length(separator)):length(files(k).folder));
        subfoldername=strcat(foldername,subfolder);
        if ~strcmp(subfolderold,subfolder)
           mkdir(subfoldername)
        end
        subfolderold = subfolder;
        copyfile(strcat(files(k).folder,'\',files(k).name),subfoldername)
    end
end

%% Back to Matlab function folder 
cd ..
cd Matlab_functions
