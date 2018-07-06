%  Copyright by Shipeng Xie ,2018.1
% All rights reserved.


% Firstly,Please install Matconvnet



% clear; clc;
format compact;

addpath(fullfile('data','utilities'));
%folderTest  = fullfile('data','Test','lungcancer'); %%% test dataset
folderTest  = fullfile('data','Test','Set20'); %%% test dataset

useGPU      = 0;

load('net-epoch-29-120fbp-512.mat');
net=dagnn.DagNN.loadobj(net);
net.mode = 'test' ;
if useGPU
  net.move('gpu') ;
end

%%% read images
ext         =  {'*.dcm'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end
	
%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
SCALE = 512;
% for i = 1:length(filePaths)
 for i = 1:1
    %%% read images
    label1 = dicomread(fullfile(folderTest,filePaths(i).name));
	label=imresize(label1,[SCALE SCALE]);
    label = single(CBFbp(label,720));
    figure,imshow(label,[-74,1456],'border','tight','initialmagnification','fit');
  	set (gcf,'Position',[0,0,512,512]);
    axis normal;
% 	imcontrast
    
    input = CBFbp(label1,120);
    input = single(input);
    figure,imshow(input,[-74,1456],'border','tight','initialmagnification','fit');
  	set (gcf,'Position',[0,0,512,512]);
    axis normal;

    if useGPU
        input = gpuArray(input);
    end
    
    net.eval({'input', input}) ;
    net.mode = 'test';
    % obtain the Net output
    res = net.vars(net.getVarIndex('x64')).value ;   
    output = input - res;
    figure,imshow(output,[-74,1456],'border','tight','initialmagnification','fit');
  	set (gcf,'Position',[0,0,512,512]);
    axis normal;
% 	imcontrast
   
    %%% convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    
    %%% calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(label,output,0,0);
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
    if ~exist('Set20_output_120FBP','file') 
           mkdir('Set20_output_120FBP');
    end
    dicomwrite(int16(output),fullfile(['Set20_output_120FBP/','PSNR=',num2str(PSNRCur),'SSIM=',num2str(SSIMCur),filePaths(i).name]));

    if ~exist('Original_FBP512','file') 
           mkdir('Original_FBP512');
    end
    dicomwrite(int16(label),fullfile(['Original_BP512/',filePaths(i).name]));

end

disp([mean(PSNRs),mean(SSIMs)]);



