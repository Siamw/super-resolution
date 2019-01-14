## Matlab 평가판(trial-use) download (ID당 30일 사용가능)


# RGB to HSI (VDSR)

## VDSR data 처리 code 분석 (code에 주석)




**generate_train.m**  
data를 **ycbcr로 만들고**, 상하좌우 filp, rotate, downsize하고, scale별로 resize 할 수 있도록 crop 뒤 imresize를 bicubic으로 진행한다.  이 때, stride크기로 data를 나누어 sub화시켜 저장하고, 순서를 섞어 각각 data(bicubic image)와 label image(hr image)로 저장한다. 이후 이 data를 HDF5로 저장한다.(train.h5)
  
**->HSI로 만들도록 바꾸어 train 시킨다.**  

~~~
clear;close all;

folder = 'path/to/train/folder';

savepath = 'train.h5';
size_input = 41;
size_label = 41;
stride = 41;

%% scale factors
scale = [2,3,4];
%% downsizing
downsizes = [1,0.7,0.5];

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

for i = 1 : length(filepaths)
for flip = 1: 3
for degree = 1 : 4
for s = 1 : length(scale)
for downsize = 1 : length(downsizes)
image = imread(fullfile(folder,filepaths(i).name));

if flip == 1
% image flip top bottom one time
image = flipdim(image ,1);
end
if flip == 2
% image flip left right one time
image = flipdim(image ,2);
end
% degree,, rotate
image = imrotate(image, 90 * (degree - 1));
% downsize
image = imresize(image,downsizes(downsize),'bicubic');

% if dim 3 size = 3 ( if, n x m x 3 )
if size(image,3)==3
% rgb to ycbcr ,,,, if dim is not 3, we can make it hsi, use i instead of ycbcr Y data
image = rgb2ycbcr(image);
image = im2double(image(:, :, 1));

% at modcrop.m, resize image as divided into scale
im_label = modcrop(image, scale(s));
% hei, wid = size that is cropped
[hei,wid] = size(im_label);

% im_label * 1/scale using bicubic interpolation and make it same size again = resolution down
im_input = imresize(imresize(im_label,1/scale(s),'bicubic'),[hei,wid],'bicubic');

filepaths(i).name

% for loop
% from 1 to hei-size_input(41)+1 , increase stride
% hei-size_input+1 means the last start of loop (not that important)
% it means from start to end, with stride

for x = 1 : stride : hei-size_input+1
for y = 1 :stride : wid-size_input+1
% im_input = image that downsampled by bicubic
subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
% im_label = image that modcropped
subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
% make sub images

% from 0 to lenth(filepaths) * 3 * 4 * length(scales) * length(Downsize)
count=count+1;

% store sub images

data(:, :, 1, count) = subim_input;
label(:, :, 1, count) = subim_label;
end
end


end
end    
end
end
end
end


% random sequence
order = randperm(count);
% change data and label order as randperm(count)
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
batchno
last_read=(batchno-1)*chunksz;
batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
created_flag = true;
totalct = curr_dat_sz(end);
end

h5disp(savepath);

~~~


**generate_test_mat.m**   
YCbCr로 data의 mat 생성  
~~~
clear;close all;
%% settings
folder = 'Set5';

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

scale = [2, 3, 4];

for i = 1 : length(filepaths)
im_gt = imread(fullfile(folder,filepaths(i).name));
for s = 1 : length(scale)
% crop as scale
im_gt = modcrop(im_gt, scale(s));
% make dataset double
im_gt = double(im_gt);
% ground truth ycbcr
im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
%im_gt_i = im_gt[][] * 255.0;

%low
im_l_ycbcr = imresize(im_gt_ycbcr,1/scale(s),'bicubic');
%bicubic
im_b_ycbcr = imresize(im_l_ycbcr,scale(s),'bicubic');

%low Y
im_l_y = im_l_ycbcr(:,:,1) * 255.0;
%low RGB
im_l = ycbcr2rgb(im_l_ycbcr) * 255.0;

% bicubic Y
im_b_y = im_b_ycbcr(:,:,1) * 255.0;
% bicubic rgb
im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;

% last = exept name after dot
last = length(filepaths(i).name)-4;
%sprintf
filename = sprintf('Set5_mat/%s_x%s.mat',filepaths(i).name(1 : last),num2str(scale(s)));
% save at filename, these things
save(filename, 'im_gt_y', 'im_b_y', 'im_gt', 'im_b', 'im_l_ycbcr', 'im_l_y', 'im_l');
end
end

~~~


**main_vdsr.py**  
train시, dataset은 generate_train.m으로 만들어진 train.h5 file로 사용된다.  
만들어진 dataset으로 model에 따라 train을 진행한다.  



**eval.py**  
generate_test_mat.m에서 만들어낸 metadata로, eval.py를 돌려 PSNR예측한다. (즉 eval.py를 돌리기 이전, 아마 main_vdsr.py에서 metadata를 만들어내는 코드가 있을 것이다.)  

_mat file의 groundtruth와 bicubic의 Y성분 metadata를 받아와서 float형태로 바꾸어 저장한다.  
이후 두개(gt and b)로 bicubic의 PSNR을 계산하여 bicubic psnr의 avg를 구한다.  

이 때, im_input은 bicubic y image를 255로 나눈 것으로, 이를 model에 넣고 돌려 HR을 생성해낸다.   

im_h_y는 high resolution Y 값으로, model(im_input = im_n_y/255)의 0번째 값이다.  
이후 다시 255를 곱해주고, 0 미만 255 초과 값은 0또는 255로 만들어준다. & im_h_y = im_h_y[0,:,;]  

이 때, groundtruth와 model을 거쳐 나온 hr을 가지고 PSNR을 계산한다.  


**demo.py**  
Set5에서 gt, b data를 받아온다.  
이 때, im_input은 im_b_y/255 이고, model에 집어넣어 data를 구한다.(자세한 설명은 아래 eval.py)  
이후 YCbCr 을 RGB로 변환하여 결과를 저장한다.   

**modcrop.m**  
~~~
function imgs = modcrop(imgs, modulo)
% make sz that can divided into modulo
% reason is that can be resize to 1/scale
% it is used in training set, so data loss doesnt matter
if size(imgs,3)==1
sz = size(imgs);
sz = sz - mod(sz, modulo);
imgs = imgs(1:sz(1), 1:sz(2));
else
tmpsz = size(imgs);
sz = tmpsz(1:2);
sz = sz - mod(sz, modulo);
imgs = imgs(1:sz(1), 1:sz(2),:);
end
~~~

  


function hsi = rgb2hsi(rgb)

%RGB2HSI Converts an RGB image to HSI.
%   HSI = RGB2HSI(RGB) converts an RGB image to HSI. The input image
%   is assumed to be of size M-by-N-by-3, where the third dimension
%   accounts for three image planes: red, green, and blue, in that
%   order. If all RGB component images are equal, the HSI conversion
%   is undefined. The input image can be of class double (with values
%   in the range [0, 1]), uint8, or uint16. 
%
%   The output image, HSI, is of class double, where:
%     hsi(:, :, 1) = hue image normalized to the range [0, 1] by
%                    dividing all angle values by 2*pi. 
%     hsi(:, :, 2) = saturation image, in the range [0, 1].
%     hsi(:, :, 3) = intensity image, in the range [0, 1].

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2005/01/18 13:44:59 $

% Extract the individual component images.
rgb = im2double(rgb);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);

% Implement the conversion equations.
num = 0.5*((r - g) + (r - b));
den = sqrt((r - g).^2 + (r - b).*(g - b));
theta = acos(num./(den + eps));

H = theta;
H(b > g) = 2*pi - H(b > g);
H = H/(2*pi);

num = min(min(r, g), b);
den = r + g + b;
den(den == 0) = eps;
S = 1 - 3.* num./den;

H(S == 0) = 0;

I = (r + g + b)/3;

% Combine all three results into an hsi image.
hsi = cat(3, H, S, I);




~[havetochange](./images/havetochange/jpeg)
