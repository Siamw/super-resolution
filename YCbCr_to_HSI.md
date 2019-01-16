# VDSR code에서 YCbCr to HSI
기존의 VDSR은 RGB를 YCbCr로 바꾸어 Y성분의 residual로 super-resolution을 진행하지만,  
HSI의 I 성분을 이용하여 super-resolution을 진행해보려 한다.  


## generate_train.m

H,S,I성분 중 I 성분만으로 train을 시키기 때문에, H,S 성분은 필요하지 않다.  
따라서 복잡하게 RGB를 HSI 성분 전체로 변환할 필요 없이,  
간단한 식 I = (R+G+B)/3 공식을 사용하여 I 성분만을 구해주면 된다.   

기존 코드   
~~~
% rgb to ycbcr ,,,, if dim is not 3, we can make it hsi, use i instead of ycbcr Y data
image = rgb2ycbcr(image);
image = im2double(image(:, :, 1));
~~~
이 부분은, RGB를 YCbCr로 바꾸어 image = Y 성분이 되도록 만들어 주는 코드이다.  
  
image = I 성분이 되도록 코드를 변경하였다.  
변경한 코드  
~~~
rgb = im2double(image);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);

image = (r+g+b)/3
~~~


## generate_test_mat.m

 HSI의 bicubic data와 ground truth 사이의 PSNR test를 위한 mat을 만들어주는 코드이다.
 전체적으로 바뀌었기 때문에 코드 전체를 첨부하였다.  
   
일단 전체적으로 ycbcr -> hsi, y -> i 로 바꾸어주었고,  
실제로 color space를 변경하기 위해서 rgb2hsi, hsi2rgb 함수를 따로 만들어서 사용하였다.  
 
 기존 코드   
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

변경한 코드  
~~~
clear;close all;
%% settings
folder = '../Set5';

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

scale = [2, 3, 4];

for i = 1 : length(filepaths)
im_gt = imread(fullfile(folder,filepaths(i).name));

for s = 1 : length(scale)
% crop as scale
im_gt = modcrop(im_gt, scale(s));

% make dataset double
im_gt = double(im_gt);

R=im_gt(:,:,1);
G=im_gt(:,:,2);
B=im_gt(:,:,3);
gt_I=(R+G+B)/3;

% ground truth ycbcr
%im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
im_gt_hsi = rgb2hsi(im_gt);

%im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
im_gt_i = gt_I;

%low
im_l_hsi = imresize(im_gt_hsi,1/scale(s),'bicubic');
%bicubic
im_b_hsi = imresize(im_l_hsi,scale(s),'bicubic');

%low Y
im_l_i = im_l_hsi(:,:,3);
%low RGB
im_l = hsi2rgb(im_l_hsi);

% bicubic Y
im_b_i = im_b_hsi(:,:,3)
% bicubic rgb
im_b = hsi2rgb(im_b_hsi);

% last = exept name after dot
last = length(filepaths(i).name)-4;
%sprintf
filename = sprintf('../Set5_mat/%s_x%s.mat',filepaths(i).name(1 : last),num2str(scale(s)));
% save at filename, these things
save(filename, 'im_gt_i', 'im_b_i', 'im_gt', 'im_b', 'im_l_hsi', 'im_l_i', 'im_l');
end
end



%% hsi 2 rgb
function outpict=hsi2rgb(inpict)
%   HSI2RGB(INPICT)
%       undoes an HSI conversion from (RGB2HSI)
%
%   INPICT is an image of class double wherein
%       H \in [0 360)
%       S \in [0 1]
%       I \in [0 1]
%
%   Return type is double, scaled [0 1]
%
%   See also: RGB2HSI
% keep in mind that not all color points in the cylinder
% specified by the extents listed above are still in the sRGB cube
% keep in mind that HSI is not a biconic model
% the isosurfaces of S (cone) and I (plane) intersect beyond the cube
% for large S and I
H=inpict(:,:,1);
S=inpict(:,:,2);
I=inpict(:,:,3);
H=H-360*floor(H/360);

% sector masks
a=H<120;
b=H<240 & ~a;
c=~a & ~b;
R=zeros(size(H));
G=zeros(size(H));
B=zeros(size(H));
B(a)=I(a).*(1-S(a));
R(a)=I(a).*(1+S(a).*cos(H(a).*(pi/180))./cos((60-H(a))*(pi/180)));
G(a)=3*I(a)-R(a)-B(a);
H(b)=H(b)-120;
R(b)=I(b).*(1-S(b));
G(b)=I(b).*(1+S(b).*cos(H(b).*(pi/180))./cos((60-H(b))*(pi/180)));
B(b)=3*I(b)-R(b)-G(b);
H(c)=H(c)-240;
G(c)=I(c).*(1-S(c));
B(c)=I(c).*(1+S(c).*cos(H(c).*(pi/180))./cos((60-H(c))*(pi/180)));
R(c)=3*I(c)-G(c)-B(c);
outpict=cat(3,R,G,B);
end

%% rgb 2 hsi
function outpict=rgb2hsi(inpict)
%   RGB2HSI(INPICT)
%       performs an HSI conversion on an rgb image
%
%   INPICT is an rgb image of class uint8 or double
%
%   Return type is double, scaled as such:
%       H \in [0 360)
%       S \in [0 1]
%       I \in [0 1]
%
%   See also: HSI2RGB
inpict = im2double(inpict)
R=inpict(:,:,1);
G=inpict(:,:,2);
B=inpict(:,:,3);
a=0.5*(2*R-G-B);
b=sqrt(3)/2*(G-B);

H=zeros(size(R));
S=zeros(size(R));
I=(R+G+B)/3;
mn=min(inpict,[],3);
nz=I~=0;
S(nz)=1-mn(nz)./I(nz);
S(~nz)=0;
H(nz)=atan2(b(nz),a(nz))*(180/pi);
H(H<0)=H(H<0)+360;
H(~nz)=0;
outpict=cat(3,H,S,I);
end

~~~


## eval.py
코드의 큰 변화는 없고, 변수명 정도만 변경하였다. 코드는 생략하겠다.  
* YCbCr 사용하였을 때의 evaluation  
![YCbCrEval](./images/ycbcr_eval.png)  
  
* YCbCr 사용하였을 때의 evaluation  
![HSIEval](./images/hsi_eval.png)


## demo.py
generate_test_mat.m와 같이 RGB2HSI, HSI2RGB를 만들어주어야 한다.  
위에서 matlab으로 만들어 사용하였으나, python버전은 없어 직접 만들어주었다.  
  
이미지를 불러올 때, 바로 HSI space로 불러올 수 없어서, 기존 YCbCr을 사용할때와 다르게  
이미지를 일단 RGB로 만들어 준 뒤 HSI로 변경하였다.   
  
함수는 다음과 같다.  
* RGB2HSI
~~~
def RGB2HSI(rgb)
rgb = rgb.astype(float);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);

h= math.acos((((r-g)+(r-b))*(0.5))/(((r-g)**2+(r-b)*(g-b))**(0.5)))

if b<=g :
H = h;
else :
H = 360-h;

S= 1 - 3/(r+g+b)*min(r,g,b)

I=(R+G+B)/3;

HSI = zeros(size(rgb))
HSI(:,:,1) = H;
HSI(:,:,2) = S;
HSI(:,:,3) = I;

return HSI
~~~

* HSI2RGB
~~~
def HSI2RGB(h_i,b_hsi) # H,S 성분은 im_b_hsi와 동일하다.
# 즉 변경된 i 성분을 가지고 super resolution을 진행하는 것!
h = b_hsi(:,:,1);
s = b_hsi(:,:,2);
i = h_i;

if h >=0 and h<120 :
B = i(1-s);
R = i(1 + (  ( s * math.cos(h) ) / ( math.cos(60-h) ) ) );
G = 3*i - (R+B);
elif h>=120 and h<240 :
h = h-120;
R = i(1-s);
G = i(1 + (  ( s * math.cos(h) ) / ( math.cos(60-h) ) ) );
B = 3*i - (R+G)
elif h>=240 and h<=360 :
h = h-240;
G = i(1-s);
B = i(1 + (  ( s * math.cos(h) ) / ( math.cos(60-h) ) ) );
R = 3*i - (G+B);

RGB = zeros(size(b_hsi))
RGB(:,:,1) = R;
RGB(:,:,2) = G;
RGB(:,:,3) = B;
return
~~~

아직 코드를 돌려보는중이라 결과는 없다. 
