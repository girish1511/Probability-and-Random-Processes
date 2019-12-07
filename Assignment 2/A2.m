%% Training
kt=zeros(512,512,10);
for i=1:10
    img=im2double(imread(strcat('Train/',int2str(i),'.gif')));
    n=imnoise(img,'gaussian',0,0.01)-img;
    N=fft2(n);          %PSD of noise
    F=fft2(img);        %PSD of original image
    kt(:,:,i)=(abs(N).^2)./(abs(F).^2);
end

%Finding average of ratios of PSD of noise and image
k=zeros(512,512);
for i=1:10
    k=k+kt(:,:,i);
end
k=k/10;

%% Testing
%Reading Test Image
e=zeros(5,1);
r=e;
x=e;
for i=1:5
im=im2double(imread(strcat('Test/',int2str(i),'.gif')));
[m,n,d]=size(im);
% subplot(1,3,1);
% imshow(im);
% title('Original')

%Applying Gaussian Blur
h=fspecial('gaussian',[5 5],10);
im_blur=imfilter(im,h);
% subplot(1,3,2);
% imshow(im_blur);
% title('Gaussian blur')

%Adding White Gaussian Noise
imn=imnoise(im_blur,'gaussian',0,0.01);
% subplot(1,3,3);
% imshow(imn);
% title('Gaussian blur + Gaussian noise')

% Transforms
figure
%Fourier transform of point spread function
H=fft2(h,m,n);
Hf=abs(H)/max(max(abs(H)));
subplot(1,2,1)
imshow(fftshift(Hf));
title('Fourier Transform of point spread function');

%Fourier transform of degraded image
G=fft2(imn);
Gf=abs(G)*255/max(max(abs(G)));
subplot(1,2,2)
imshow(fftshift(Gf));
title('Fourier Transform of degraded image');
% Applying Wiener Filter

%Calculating Wiener filter
Hc=conj(H);
W=Hc./((abs(H).^2)+k);

%Taking inverse fourier transform and obtaining restored image
Fr=(G.*W);
f=ifft2(Fr);
fr=abs(f)/max(max(abs(f)));
% imshow(fftshift(abs(Fr)));
% title('Fourier Transform of restored image');

%Displaying images
figure
subplot(2,2,1)
imshow(fr);
title('Restored image');
subplot(2,2,2)
imshow(im)
title('Orignal');
subplot(2,2,3)
imshow(imn)
title('Noise+Blur')
subplot(2,2,4)
imshow(im_blur)
title('Blur');

%Mean Square Error
e(i)=immse(abs(f),im);
%Peak Signal to Noise Ratio of restored image
r(i)=psnr(abs(f),im);
%Peak Signal to Noise Ratio of corrupt image
x(i)=psnr(imn,im);

% fprintf('\nMean Square Error = %f',e);
% fprintf('\nPeak Signal to Noise Ratio restored image= %f',r);
% fprintf('\nPeak Signal to Noise Ratio corrupt image= %f',x);
% fprintf('\nIncrement in psnr= %f\n',r-x);
end
%% Results

plot(1:5,r);
grid on
xlabel('Image No')
ylabel('PSNR')
title('PSNR of Restored Image')

figure
plot(1:5,r-x);
grid on
xlabel('Image No')
ylabel('Increment in PSNR')
title('Increment in PSNR')

fprintf('\n\tImage No\tPSNR(Restored image)\tPSNR(Corrupt image)\tIncrement in PSNR\n')
for i=1:5
    fprintf('\t%d\t\t%f\t\t%f\t\t%f\n',i,r(i),x(i),r(i)-x(i));
end
grid on