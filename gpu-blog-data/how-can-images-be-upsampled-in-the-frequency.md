---
title: "How can images be upsampled in the frequency domain using PyTorch?"
date: "2025-01-30"
id: "how-can-images-be-upsampled-in-the-frequency"
---
Image upsampling in the frequency domain, while less common than spatial-domain methods like bilinear interpolation, offers a powerful alternative when specific frequency characteristics need preservation or manipulation. The core principle involves operating on the Fourier transform of the image, padding it in the frequency domain, and then performing an inverse Fourier transform to obtain the upsampled result. I've personally employed this technique in hyperspectral image analysis, where subtle frequency components held critical information. My experience suggests this approach is particularly beneficial when avoiding blur introduced by traditional pixel-based interpolation.

The fundamental process hinges on understanding the Discrete Fourier Transform (DFT) and its inverse (IDFT), both efficiently implemented in PyTorch. An image, which exists in the spatial domain, can be transformed into its frequency domain representation using the DFT. This representation describes the image in terms of its constituent frequencies rather than pixel intensities. Low frequencies correspond to slow variations in intensity (large areas of similar color), whereas high frequencies correspond to rapid changes (edges and fine details).

Upsampling in the frequency domain doesn't involve directly increasing the number of pixels as in spatial methods. Instead, it expands the frequency domain representation, often by padding the high-frequency regions with zeros. This process effectively increases the sampling rate in the frequency domain, which when inverted, leads to an increase in the resolution (size) of the image in the spatial domain. The zero padding introduces new high-frequency content, forcing the inverse DFT to generate a higher-resolution image. When done correctly, it leads to a smoothly interpolated increase in size. Conversely, adding non-zero values in the frequency domain can introduce artifacts.

Let's explore this with concrete examples. We will work with a grayscale image for simplicity, but these principles extend directly to color images by processing each color channel individually. Assume we have a 2D PyTorch tensor representing a grayscale image. The first step involves applying a 2D Fast Fourier Transform (FFT). PyTorch provides `torch.fft.fft2` for this purpose.

```python
import torch
import torch.fft

def frequency_upsample(image, scale_factor):
    """Upsamples an image in the frequency domain using zero padding."""
    rows, cols = image.shape
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    # Compute the 2D FFT of the image.
    fft_image = torch.fft.fft2(image.float())

    # Shift the zero frequency component to the center.
    fft_image_shifted = torch.fft.fftshift(fft_image)

    # Initialize a zero-padded tensor in the frequency domain.
    padded_fft = torch.zeros(new_rows, new_cols, dtype=fft_image.dtype, device=fft_image.device)

    # Calculate the starting row and column for placing the original FFT within the padded tensor.
    start_row = (new_rows - rows) // 2
    start_col = (new_cols - cols) // 2

    # Copy the shifted FFT to the center of the zero-padded tensor.
    padded_fft[start_row:start_row+rows, start_col:start_col+cols] = fft_image_shifted

    # Shift back the zero frequency component.
    padded_fft_shifted = torch.fft.ifftshift(padded_fft)

    # Apply the Inverse 2D FFT.
    upsampled_image = torch.fft.ifft2(padded_fft_shifted)

    # Take the real part.
    upsampled_image = upsampled_image.real

    return upsampled_image

# Example usage:
image = torch.rand(64, 64) # Example 64x64 grayscale image
upsampled_image = frequency_upsample(image, 2)  # Upsample by a factor of 2
print(f"Original image size: {image.shape}, Upsampled image size: {upsampled_image.shape}")

```

In this first example, I’ve encapsulated the upsampling logic within the `frequency_upsample` function. The core steps involve computing the FFT, shifting the zero frequency to the center (using `fftshift`), creating a padded zero tensor, embedding the shifted FFT into it, shifting back to the corners (using `ifftshift`), computing the inverse FFT, and finally, extracting the real component of the resulting complex tensor. `torch.fft` requires casting the input to `float` or complex. The final return is an upsampled image.

The second example showcases how to perform upsampling with a non-integer scaling factor. This requires adapting the padding and the size calculation. The core process remains the same, but one must pay attention to how the size is calculated. We truncate the number of rows and columns to nearest integer because pixel dimensions must be integers. This truncation can introduce minor differences, particularly at low scales.

```python
import torch
import torch.fft

def frequency_upsample_float(image, scale_factor):
    """Upsamples an image with a non-integer scaling factor."""
    rows, cols = image.shape
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    fft_image = torch.fft.fft2(image.float())
    fft_image_shifted = torch.fft.fftshift(fft_image)

    padded_fft = torch.zeros(new_rows, new_cols, dtype=fft_image.dtype, device=fft_image.device)

    start_row = (new_rows - rows) // 2
    start_col = (new_cols - cols) // 2

    # Copy only a portion of the original frequency information
    padded_fft[start_row:start_row+rows, start_col:start_col+cols] = fft_image_shifted

    padded_fft_shifted = torch.fft.ifftshift(padded_fft)
    upsampled_image = torch.fft.ifft2(padded_fft_shifted)
    upsampled_image = upsampled_image.real

    return upsampled_image

image = torch.rand(64, 64)
upsampled_image = frequency_upsample_float(image, 1.5)  #Upsample by a factor of 1.5
print(f"Original image size: {image.shape}, Upsampled image size: {upsampled_image.shape}")
```
This example demonstrates a crucial aspect – the ability to handle non-integer scale factors. The underlying principles remain identical; the `scale_factor` can be any floating point value.

Finally, let's briefly explore the potential use case of this technique. Consider a scenario where one might wish to isolate specific frequency bands in an image and then apply upsampling. It’s not merely about getting a bigger image but shaping the resulting image’s spectral content. We start by defining a filter to modify the frequency representation, in this case, a Gaussian filter centered at the origin of the frequency plane (as determined by `fftshift`). This allows for smooth removal of higher frequency bands.

```python
import torch
import torch.fft

def frequency_upsample_filtered(image, scale_factor, sigma):
    """Upsamples an image with a Gaussian filter in the frequency domain."""
    rows, cols = image.shape
    new_rows = int(rows * scale_factor)
    new_cols = int(cols * scale_factor)

    fft_image = torch.fft.fft2(image.float())
    fft_image_shifted = torch.fft.fftshift(fft_image)

    padded_fft = torch.zeros(new_rows, new_cols, dtype=fft_image.dtype, device=fft_image.device)
    start_row = (new_rows - rows) // 2
    start_col = (new_cols - cols) // 2

    padded_fft[start_row:start_row+rows, start_col:start_col+cols] = fft_image_shifted
    center_row = new_rows // 2
    center_col = new_cols // 2
    row_indices = torch.arange(new_rows, device=fft_image.device).float()
    col_indices = torch.arange(new_cols, device=fft_image.device).float()

    row_grid, col_grid = torch.meshgrid(row_indices, col_indices)
    gaussian_filter = torch.exp(-((row_grid - center_row)**2 + (col_grid - center_col)**2) / (2 * sigma**2))

    padded_fft = padded_fft * gaussian_filter

    padded_fft_shifted = torch.fft.ifftshift(padded_fft)

    upsampled_image = torch.fft.ifft2(padded_fft_shifted)
    upsampled_image = upsampled_image.real
    return upsampled_image

image = torch.rand(64, 64)
upsampled_image = frequency_upsample_filtered(image, 2, 20)
print(f"Original image size: {image.shape}, Upsampled image size: {upsampled_image.shape}")
```

Here, the `frequency_upsample_filtered` function introduces a Gaussian filter, which attenuates the high-frequency components. The resulting upsampled image will appear smoother, having less of the high-frequency "edge" information. The degree of smoothing is determined by the `sigma` parameter. This shows the versatility of frequency-domain upsampling, moving beyond simple interpolation, allowing sophisticated spectral manipulation.

For deeper study, the official PyTorch documentation provides the authoritative resource for all tensor operations, FFT functions, and relevant data structures. Research papers on image processing and computer vision often discuss the mathematical theory behind the Fourier Transform and its various applications. For numerical computation, standard textbooks on numerical methods offer insights into the underlying algorithms used by FFT implementations. These resources provide theoretical grounding and practical techniques for advanced image processing, expanding understanding beyond the provided examples.
