---
title: "matlab how to implement a ram lak filter ramp filter in the frequency domain?"
date: "2024-12-13"
id: "matlab-how-to-implement-a-ram-lak-filter-ramp-filter-in-the-frequency-domain"
---

 so you're asking how to do a Ram-Lak filter in MATLAB using the frequency domain right? Classic image processing stuff Been there done that got the t-shirt and probably spilled coffee on it a few times let me tell you about it

Right off the bat if you're trying to sharpen an image or do some sort of image reconstruction a Ram-Lak filter is often the go to Its basically a high pass filter that boosts high frequencies relative to low frequencies Its useful in situations where you want to enhance edges or features while suppressing smooth regions Think of it like this you're trying to bring the details to the forefront while ignoring the boring parts of an image

Now why the frequency domain? Well the convolution theorem is your best friend here In the spatial domain filtering is done by convolution a pretty expensive operation in terms of computation time But in the frequency domain it's a simple point-wise multiplication That makes it much faster especially for larger images It's like transforming a marathon into a short sprint it saves time and energy

Here is how I have done it in the past with varying levels of success I had this one project years ago where I was trying to recover some data from what I thought was low quality medical imaging scans I mean those things can be a mess so the Ram-Lak filter was actually perfect for that I think I overused it though my colleagues ended up thinking I was a bit over the top with the edges being really sharp but hey I was learning I also tried it with seismic data once which was another interesting case it gave me some weird noises but again it's all part of the process you learn something new every time

Let's dive into the MATLAB code cause that’s what you really want

First here is the basic idea for a 1D signal This is a foundation for the 2D case which we'll cover next

```matlab
function ramLakFilter1D(signal)

    N = length(signal);
    frequencyAxis = (-N/2:N/2-1) / N;
    ramLakKernel = abs(frequencyAxis);

    signalFourier = fftshift(fft(signal));
    filteredSignalFourier = signalFourier .* ramLakKernel;
    filteredSignal = ifft(ifftshift(filteredSignalFourier));

    figure;
    subplot(2,1,1);
    plot(1:N,signal);
    title('Original signal');

    subplot(2,1,2);
    plot(1:N,real(filteredSignal));
    title('Filtered signal');

end
```

So what are we doing here? we are taking a 1D signal you give the function first then create a frequency axis that goes from -05 to 05 We then create the Ram-Lak kernel which is simply the absolute value of the frequency axis That's the core of the filter It's zero at zero frequency and increases linearly with frequency Then we move to the frequency domain with `fft` shift to put the zero frequency component at the center multiply with our kernel then back to the signal domain with `ifft` we shift again to fix the frequency ordering finally we visualize both signals to see the change It should make the signal a bit more edgy for lack of better word

Now that was a simple 1D version lets move to a 2D image which is probably more relevant to your needs

```matlab
function ramLakFilter2D(image)

    [rows, cols] = size(image);
    [x,y] = meshgrid(-cols/2:cols/2-1,-rows/2:rows/2-1);
    frequencyAxis = sqrt(x.^2+y.^2)/(max(cols,rows)/2);
    ramLakKernel = frequencyAxis;

    imageFourier = fftshift(fft2(image));
    filteredImageFourier = imageFourier .* ramLakKernel;
    filteredImage = ifft2(ifftshift(filteredImageFourier));

    figure;
    subplot(1,2,1);
    imshow(image,[]);
    title('Original Image');

    subplot(1,2,2);
    imshow(real(filteredImage),[]);
    title('Filtered Image');

end
```
This function takes an image we calculate the meshgrid for a 2D frequency axis similar to the 1D signal but now it’s a 2D grid and then calculates the radial frequency and uses it to form our kernel the kernel increases linearly with the radial frequency just like in 1D and this is the real Ram-Lak filter function You then perform the Fourier transform on the image shift it to center multiply with the filter kernel and go back to the spatial domain using the inverse Fourier transform `ifft2` display both images and you are done

A couple of notes about the code above

First the division by `max(cols,rows)/2` in the 2D case normalizes the frequency axis to have values within a 0-1 range it's a good habit to get into so you're not dealing with very big values Second notice the `real` part after `ifft` and `ifft2` since numerical errors could lead to small imaginary components we do that to get the actual signal and the image back without visual artifacts This is very common when dealing with FFT and IFFT operations

 time for a slight aside I once tried using this filter on a cat picture thinking it would make it look super cool it just looked weird not cool at all I guess cats are not meant to be sharp Anyway lets continue

Now the basic Ram-Lak filter we just implemented can be sensitive to high frequency noise in the image because it’s a high pass filter that amplifies high frequencies Now if you want to reduce that sensitivity you can add a windowing function that tapers off the high frequency response This is pretty useful if you have a very noisy image data

Here is an example of a windowed version:

```matlab
function windowedRamLakFilter2D(image, cutoffFrequency)

    [rows, cols] = size(image);
    [x,y] = meshgrid(-cols/2:cols/2-1,-rows/2:rows/2-1);
    frequencyAxis = sqrt(x.^2+y.^2)/(max(cols,rows)/2);
    ramLakKernel = frequencyAxis;

    % Hamming Window to taper high frequencies
    hammWindow = hamming(length(frequencyAxis));
    for i = 1 : length(hammWindow)
        hammWindow(i) = (0.54 - 0.46 * cos(2 * pi * i / (length(hammWindow)-1)) );
    end
    cutoffIndex = round(cutoffFrequency*(length(frequencyAxis)-1));
    window = ones(size(frequencyAxis));
    if cutoffIndex < length(window)
        window(cutoffIndex:end)= 0;
        window = window .* hammWindow';
    end
    ramLakKernelWindowed = ramLakKernel .* window;


    imageFourier = fftshift(fft2(image));
    filteredImageFourier = imageFourier .* ramLakKernelWindowed;
    filteredImage = ifft2(ifftshift(filteredImageFourier));


    figure;
    subplot(1,2,1);
    imshow(image,[]);
    title('Original Image');

    subplot(1,2,2);
    imshow(real(filteredImage),[]);
    title('Filtered Image');

end
```
This is similar to the previous example but now it has a cutoff frequency and hamming window implementation First we calculate the ram-lak filter and then calculate a 1D hamming window Then multiply that to the ram-lak filter to get a new windowed version and then we use that filter for our image This limits the amplification of high frequency and minimizes the noise in the image

The `cutoffFrequency` here is a value between 0 and 1 and it dictates at what frequency we start tapering the kernel. This is a good way to deal with noise

Now lets talk about resources if you want to get into this properly

If you want to dive deeper into image processing and filters you cannot skip Digital Image Processing by Rafael C Gonzalez and Richard E Woods This book is a classic and it covers everything you need about image processing including the theory and implementation of all kinds of filters A lot of the stuff I’ve learned came from that book It's like the bible of image processing for real

For a more mathematical and signal processing oriented approach you could check out Oppenheim and Schafer’s Discrete-Time Signal Processing It covers the foundations of discrete signals and systems that is very helpful for understanding how these filters work at a deeper level

Also you might also want to check out articles in the IEEE Transactions on Image Processing journal it has a lot of cutting edge research and papers related to image processing algorithms

 I think that covers it Feel free to ask if there’s anything else you need it’s all a learning process that never ends and we all are there in our own way good luck with your image processing adventure
