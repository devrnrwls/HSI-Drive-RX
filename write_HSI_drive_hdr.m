function write_HSI_drive_hdr(headername, format, precision, sorted)

% Write ENVI header for HSI-Drive files

% Author: Koldo Basterretxea
% Digital Eletronics Design Group (GDED)
% University of the Basque Country (UPV/EHU)
% 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bands = [888.48, 897.66, 879.13, 869.11, 956.11, 795.96, 807.63, 783.76,...
 770.58, 679.14, 744.48, 757.92, 731.88, 718.30, 692.90, 928.81, 936.25,...
 920.60, 912.17, 950.39, 848.55, 858.95, 838,08, 826.77, 944.49];

if sorted
    [bands,ind]=sort(bands);
end

if strcmp(precision,'uint16')
    datatype = '12';
else
    datatype = '5';
end

fid = fopen(headername,'wt');

fprintf(fid, 'ENVI\n');
fprintf(fid,'description = {CMV2K-SSM5x5-600_1000-5.5.9.9\n');
fprintf(fid,'sensor id = 5.5.9.9\n');
fprintf(fid,'demosaiced  = YES\n');
fprintf(fid,'median filtered  = YES\n');
fprintf(fid,'spectral corrected = NO\n');
fprintf(fid,'band normalization = YES\n');
fprintf(fid,'spatial resampling = NO }\n');
fprintf(fid,'file type = ENVI\n');
fprintf(fid,'sensor type = MOSAIC\n');
fprintf(fid,'%s%s\n','interleave = ',format);
fprintf(fid,'samples = 409\n');
fprintf(fid,'lines = 216\n');
fprintf(fid,'bands = 25\n');
fprintf(fid,'default bands = { 5, 12, 21}\n');
fprintf(fid,'header offset = 0\n');
fprintf(fid,'%s%s\n','data type = ',datatype);
fprintf(fid,'byte order = 0\n');
fprintf(fid,'x start = 0\n');
fprintf(fid,'y start = 0\n');
fprintf(fid,'max_value = 4095\n');
fprintf(fid,'Wavelength = {\n');
fprintf(fid,'%.2f,\n',bands(1:24));
fprintf(fid,'%.2f\n',bands(25));
fprintf(fid,'}\n');

fclose(fid);