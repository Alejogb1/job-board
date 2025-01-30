---
title: "How can PyTorch be used to adjust highlights and shadows?"
date: "2025-01-30"
id: "how-can-pytorch-be-used-to-adjust-highlights"
---
Implementing localized image adjustments, specifically highlights and shadows manipulation, within PyTorch requires a different approach than global transformations. Directly modifying pixel values within the image tensor, while possible, ignores the underlying spatial context that defines highlights and shadows. Instead, we leverage convolutions with learnable or predetermined kernels to extract information about local luminance patterns, and then apply a transformation based on this information. My experience has taught me that effective control relies on a combination of spatial filtering and careful non-linear mapping.

**Understanding the Problem:**

Highlight and shadow manipulation centers on adjusting the tonal values of an image based on their relative brightness within local regions. Areas with high luminance are considered potential highlights; areas with low luminance are considered potential shadows. Global adjustments, such as simple brightness or contrast, modify the entire image uniformly, impacting both highlights and shadows similarly. To achieve selective manipulation, we must first analyze the image's local context to determine which areas should be considered highlights and which shadows. Then, a mapping function is applied based on the location. This local analysis is where convolutions become indispensable. By using appropriately defined convolutional kernels, I've been able to extract the necessary local brightness information.

**Convolutional Approach:**

We will use convolutional layers to generate feature maps that respond strongly to either high or low luminance areas. This process is inspired by techniques that have been adopted in high-dynamic-range (HDR) imaging. The main steps I've utilized are:

1. **Local Luminance Extraction:** We convert the input RGB image into a grayscale representation. This step simplifies the process by reducing our three color channels into a single luminance channel. I have also tried different weighting of color channels to create a more representative grayscale and can confirm this is a fruitful direction for experimentation.
2. **Spatial Filtering:** Using convolutions, we apply two distinct filters (or filter sets). One filter highlights areas with higher luminance (acting as a high-pass), while the other emphasizes areas with lower luminance (approximating a low-pass response but designed to extract shadows). These can be achieved with learnable filters or pre-determined ones, like a simple Gaussian blur.
3. **Mapping Function:** The extracted highlight and shadow maps are used to modulate the original image, using a nonlinear mapping, such as a power law (gamma) transformation or an S-curve. The nonlinear mapping provides a way to increase or decrease the intensity of the highlights and shadows independently.

**Code Example 1: Basic Convolution and Grayscale Conversion**

This example sets up the essential components for local analysis.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_grayscale(img):
    """Converts an RGB image (B, C, H, W) to grayscale (B, 1, H, W)."""
    return 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]

class HighlightShadowAdjust(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
      super().__init__()
      self.kernel_size = kernel_size
      self.sigma = sigma

      # Define a basic Gaussian kernel to help in the extraction.
      gaussian_kernel = self.generate_gaussian_kernel(self.kernel_size, self.sigma)
      self.register_buffer('gaussian_kernel', gaussian_kernel)

      # Apply a single convolutional layer.
      self.conv = nn.Conv2d(1, 1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2, bias=False)
      self.conv.weight = nn.Parameter(self.gaussian_kernel)
      self.conv.requires_grad_(False)


    def generate_gaussian_kernel(self, size, sigma):
        x = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float)
        y = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float)
        x_grid, y_grid = torch.meshgrid(x, y)
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)


    def forward(self, img, gamma_highlights=1.2, gamma_shadows=0.8):
      """Adjusts highlights and shadows of an image."""
      gray_img = rgb_to_grayscale(img).unsqueeze(1)
      # Gaussian blur the gray image for soft extraction.
      blurred_img = self.conv(gray_img)

      # Compute highlight and shadow masks.
      highlights = (gray_img - blurred_img).clamp(min=0) # Areas brighter than the blur
      shadows = (blurred_img - gray_img).clamp(min=0)   # Areas darker than the blur

      # Apply gamma adjustments.
      adjusted_img = torch.pow(img, gamma_highlights * highlights + gamma_shadows * shadows + 1*(1- highlights - shadows))
      return adjusted_img


# Example usage
input_image = torch.rand(4, 3, 256, 256) # Batch of 4 RGB images
adjust_layer = HighlightShadowAdjust(kernel_size=7, sigma=2.5) # Initialized gaussian kernel of size 7 and std 2.5
output_image = adjust_layer(input_image, gamma_highlights=1.3, gamma_shadows=0.7)
print(output_image.shape) # Output tensor shape
```

*Commentary:*

This example demonstrates the basic process: converting the image to grayscale, creating a Gaussian kernel, performing a convolution operation, and then generating rudimentary highlight and shadow maps by comparing the blurred and original image. These two feature maps are then used to apply gamma adjustment over the entire image. This basic code provides a framework from which other more complex and specific adjustments can be built.

**Code Example 2: Learnable Convolution and S-curve Mapping**

This example demonstrates replacing a pre-determined kernel with a learnable one, and employing an S-curve for the mapping.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rgb_to_grayscale(img):
    """Converts an RGB image (B, C, H, W) to grayscale (B, 1, H, W)."""
    return 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]


class HighlightShadowAdjustLearnable(nn.Module):
  def __init__(self, kernel_size=3):
    super().__init__()
    self.kernel_size = kernel_size

    self.conv_highlights = nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2)
    self.conv_shadows = nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2)


  def s_curve(self, x, slope=5.0):
    """Applies an S-curve mapping to the image, controlled by slope."""
    return 1 / (1 + torch.exp(-slope * (x - 0.5)))

  def forward(self, img, slope_highlights=5.0, slope_shadows=5.0):
        """Adjusts highlights and shadows of an image."""
        gray_img = rgb_to_grayscale(img).unsqueeze(1)

        highlights = self.conv_highlights(gray_img)
        shadows = self.conv_shadows(gray_img)


        # Normalize between 0 and 1, and apply s-curves
        highlights = self.s_curve(F.sigmoid(highlights), slope=slope_highlights)
        shadows = self.s_curve(F.sigmoid(shadows), slope=slope_shadows)


        # Apply highlight and shadow masks
        adjusted_img = img * (1 + highlights - shadows) # Basic addition and subtraction here for simplicity
        return adjusted_img



# Example usage
input_image = torch.rand(4, 3, 256, 256)
adjust_layer = HighlightShadowAdjustLearnable(kernel_size=5)
output_image = adjust_layer(input_image, slope_highlights=7.0, slope_shadows=2.0)
print(output_image.shape)

```

*Commentary:*

This example replaces the fixed Gaussian filter with two trainable convolutional layers. This allows the network to learn optimal kernels for extracting highlights and shadows. The output of these convolutions is then passed through an S-curve function. This allows for more nuanced adjustments. We apply two different slopes for highlights and shadows.

**Code Example 3: Complex Mapping and Multi-Stage Adjustments**

This example introduces a more complex mapping function and allows the use of separate control over how strongly highlights and shadows are adjusted.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def rgb_to_grayscale(img):
    """Converts an RGB image (B, C, H, W) to grayscale (B, 1, H, W)."""
    return 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]


class HighlightShadowAdjustAdvanced(nn.Module):
    def __init__(self, kernel_size=3):
      super().__init__()
      self.kernel_size = kernel_size

      # Using two layers for more complex control of highlights and shadows.
      self.conv_highlights = nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2)
      self.conv_shadows = nn.Conv2d(1, 1, kernel_size=self.kernel_size, padding=self.kernel_size//2)

    def mapping_function(self, x, highlight_intensity, shadow_intensity):
        """Complex nonlinear mapping function."""
        # Use intensity params to control amount of contrast.
        highlight_effect = torch.sigmoid(x) * highlight_intensity
        shadow_effect = torch.sigmoid(x) * shadow_intensity
        return 1 + highlight_effect - shadow_effect

    def forward(self, img, highlight_intensity=1.5, shadow_intensity=0.5):
        gray_img = rgb_to_grayscale(img).unsqueeze(1)

        highlights = self.conv_highlights(gray_img)
        shadows = self.conv_shadows(gray_img)


        # Normalize between 0 and 1
        highlights = F.sigmoid(highlights)
        shadows = F.sigmoid(shadows)

        # Apply the complex mapping function
        adjusted_img = img * self.mapping_function(highlights, highlight_intensity, shadow_intensity)

        return adjusted_img

# Example usage
input_image = torch.rand(4, 3, 256, 256)
adjust_layer = HighlightShadowAdjustAdvanced(kernel_size=7)
output_image = adjust_layer(input_image, highlight_intensity=1.2, shadow_intensity=0.3)
print(output_image.shape)

```

*Commentary:*

In this example, we introduce a more complex mapping function. We use an intermediate intensity parameter to control the strength of highlights and shadows. This gives more control over the final look and enables a degree of fine-tuning, something I found important in my experience. Note this code can become more complex easily with the introduction of additional mapping functions or convolutional layers.

**Resource Recommendations:**

For further study of this specific area, I would recommend reviewing material on the following:

1.  **Image Processing Fundamentals:** A solid grounding in basic image processing techniques (e.g., filtering, histograms, color spaces) is essential. Textbooks on digital image processing provide comprehensive coverage of these concepts.
2.  **Convolutional Neural Networks (CNNs):** In-depth knowledge of CNNs is crucial, including understanding convolution operations, various network architectures, and optimization methods. Books and articles on deep learning cover CNNs extensively.
3.  **High-Dynamic-Range (HDR) Imaging Techniques:** Researching HDR imaging can offer different perspectives on tone mapping and the manipulation of luminance in images. Publications from the computer graphics and image processing fields offer insights into this specific topic.
4.  **Nonlinear Mapping Functions:** Investigate various nonlinear functions used for image adjustments. Experimenting with different mapping functions beyond simple gamma transformations will aid in more subtle and creative results.
5.  **PyTorch Documentation:** A thorough understanding of PyTorch's tensor operations, neural network modules, and optimization tools is vital. Refer to the official PyTorch documentation for detailed information and examples.

By combining these core competencies, it is possible to craft efficient and customized highlight and shadow adjustment tools using PyTorch. This allows for fine-tuned control of an image's tonal range, opening opportunities for creative image manipulation.
