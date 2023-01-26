% This scripts converts the raw frame sequences in the example 'Video'
% folder into a video .avi from .jpg images

addpath Video;
fps = 11;

%% List raw images
imageNames = dir(fullfile('Video','*.bin'));
% Make a new folder to save jpeg images
mkdir(fullfile('Video','jpg'));
foldername=fullfile('Video','jpg');

% Convert .bin raw images into jpeg images and save
for i=1:length(imageNames)
    sprintf('Convertinf raw files to jpg(%d)',i);
    filename=imageNames(i).name;
    raw2jpg(filename, foldername)
end

%% List saved jpeg images and write a video file
imageNames = dir(fullfile('Video','jpg','*.jpg'));
cd Video/jpg

outputVideo = VideoWriter('20sec_drive.avi');
outputVideo.FrameRate = fps;
open(outputVideo)

for j = 1:length(imageNames)
   sprintf('Generaing video from jpeg files(%d)',i);
   img = imread(imageNames(j).name);
   writeVideo(outputVideo,img)
end
close(outputVideo)

%% Visualize video
% DriveAvi = VideoReader('20sec_drive.avi');
% k = 1;
% while hasFrame(DriveAvi)
%    mov(k) = im2frame(readFrame(DriveAvi));
%    k = k+1;
% end
% h=figure ;
% imshow(mov(1).cdata, 'Border', 'tight')
% truesize
% movie(h,mov,1,11)

