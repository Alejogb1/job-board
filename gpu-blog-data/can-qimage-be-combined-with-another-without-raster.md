---
title: "Can QImage be combined with another without raster operations?"
date: "2025-01-30"
id: "can-qimage-be-combined-with-another-without-raster"
---
Direct manipulation of `QImage` pixel data provides the most efficient method for combining images, bypassing the overhead of high-level raster operations. The key lies in accessing the raw byte arrays representing the image data and performing custom blending or compositing algorithms directly. While `QImage` provides functions like `blit()` for basic drawing, these rely on the Qt drawing system and are less performant for pixel-level combinations, particularly when complex blending is required.

My experience developing a real-time image processing application demonstrated this difference clearly. We initially employed `QPainter` with `drawImage()` calls to combine multiple layers, which proved excessively slow when dealing with high-resolution imagery and dynamic alpha changes. This led us to investigate direct pixel manipulation, resulting in a substantial performance increase. It is important to note that this method requires careful handling of image formats and memory alignment to avoid crashes or undefined behavior.

To clarify, combining two `QImage` objects without using raster operations like `QPainter`'s `drawImage()` method involves directly manipulating the pixel data buffers. This essentially means working with the underlying byte arrays representing the images. The procedure generally involves the following steps: allocating a destination `QImage` large enough to contain the result of the combination, retrieving the raw pixel data of the source images, iterating through each pixel location, applying the chosen combination or blending algorithm to the respective pixels, and finally, copying the combined pixels into the destination `QImage`'s data buffer.

This approach offers finer control over the combination process, enabling effects that may be cumbersome or inefficient to achieve with raster operations. Moreover, it provides the opportunity for optimization by exploiting SIMD instructions or multi-threading, especially beneficial for processing large images. The following three code examples will illustrate different combination scenarios using direct pixel access.

**Example 1: Simple Alpha Blending**

This first example demonstrates simple alpha blending of two `QImage` objects. One image acts as the base, and the other is blended on top based on its alpha value. This method assumes both images have the same dimensions and are in `QImage::Format_ARGB32`.

```cpp
QImage blendAlpha(const QImage &base, const QImage &overlay) {
  if (base.size() != overlay.size() || base.format() != QImage::Format_ARGB32 || overlay.format() != QImage::Format_ARGB32) {
    return QImage(); // Return an invalid image on failure
  }

  QImage result(base.size(), QImage::Format_ARGB32);
  const int width = base.width();
  const int height = base.height();
  const uchar* baseData = base.constBits();
  const uchar* overlayData = overlay.constBits();
  uchar* resultData = result.bits();

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int offset = (y * width + x) * 4;

      // Extract RGBA values from both images
      float baseB = baseData[offset];
      float baseG = baseData[offset + 1];
      float baseR = baseData[offset + 2];
      float baseA = baseData[offset + 3] / 255.0f;

      float overlayB = overlayData[offset];
      float overlayG = overlayData[offset + 1];
      float overlayR = overlayData[offset + 2];
      float overlayA = overlayData[offset + 3] / 255.0f;

      // Apply alpha blending formula
      float outA = overlayA + baseA * (1 - overlayA);
      float outR = (overlayR * overlayA + baseR * baseA * (1 - overlayA)) / outA;
      float outG = (overlayG * overlayA + baseG * baseA * (1 - overlayA)) / outA;
      float outB = (overlayB * overlayA + baseB * baseA * (1 - overlayA)) / outA;

      // Clamp values to 0-255 and store
      resultData[offset] = static_cast<uchar>(std::clamp(outB, 0.0f, 255.0f));
      resultData[offset + 1] = static_cast<uchar>(std::clamp(outG, 0.0f, 255.0f));
      resultData[offset + 2] = static_cast<uchar>(std::clamp(outR, 0.0f, 255.0f));
      resultData[offset + 3] = static_cast<uchar>(std::clamp(outA * 255.0f, 0.0f, 255.0f));
    }
  }
  return result;
}
```

This code first performs size and format checks to prevent errors. The `constBits()` function provides read-only access to the source image data, and `bits()` provides read-write access to the destination image. The nested loops iterate over every pixel. Within the loop, the RGBA values are extracted, converted to floats, blended according to the standard alpha blending formula, converted back to `uchar`, and written to the result image. The `std::clamp` function ensures the blended RGB values remain within the valid range.

**Example 2: Additive Blending**

This example showcases additive blending, where the color values of two images are added together. This often creates a 'brightening' effect and is useful in creating special lighting effects. Again, we assume both images are the same dimensions and format.

```cpp
QImage blendAdditive(const QImage &image1, const QImage &image2) {
    if (image1.size() != image2.size() || image1.format() != QImage::Format_ARGB32 || image2.format() != QImage::Format_ARGB32) {
        return QImage();
    }

    QImage result(image1.size(), QImage::Format_ARGB32);
    const int width = image1.width();
    const int height = image1.height();
    const uchar* img1Data = image1.constBits();
    const uchar* img2Data = image2.constBits();
    uchar* resultData = result.bits();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int offset = (y * width + x) * 4;
            // Extract color values from both images
            float img1B = img1Data[offset];
            float img1G = img1Data[offset + 1];
            float img1R = img1Data[offset + 2];

            float img2B = img2Data[offset];
            float img2G = img2Data[offset + 1];
            float img2R = img2Data[offset + 2];

            // Add the color values, keeping the alpha of image1
            float outB = std::clamp(img1B + img2B, 0.0f, 255.0f);
            float outG = std::clamp(img1G + img2G, 0.0f, 255.0f);
            float outR = std::clamp(img1R + img2R, 0.0f, 255.0f);

           // Store the result, keeping alpha from image 1
            resultData[offset] = static_cast<uchar>(outB);
            resultData[offset + 1] = static_cast<uchar>(outG);
            resultData[offset + 2] = static_cast<uchar>(outR);
            resultData[offset + 3] = img1Data[offset + 3];
        }
    }
    return result;
}
```

Here the procedure is similar to the alpha blending case but the color channels of the two images are directly added. Importantly, the additive blend can potentially generate values higher than 255; therefore, we use `std::clamp` to make sure that no data overflows into adjacent pixels. The alpha value from the first image is used in the result.

**Example 3: Masked Copy**

This third example demonstrates copying pixels from one image to another only if the corresponding pixel in a third ‘mask’ image has an alpha value greater than a threshold. This is commonly used for implementing various image cut-out and masking effects.

```cpp
QImage maskedCopy(const QImage &baseImage, const QImage &sourceImage, const QImage &maskImage, int alphaThreshold) {
  if (baseImage.size() != sourceImage.size() || baseImage.size() != maskImage.size() ||
      baseImage.format() != QImage::Format_ARGB32 || sourceImage.format() != QImage::Format_ARGB32 || maskImage.format() != QImage::Format_ARGB32) {
    return QImage();
  }

  QImage result(baseImage.size(), QImage::Format_ARGB32);
  const int width = baseImage.width();
  const int height = baseImage.height();
  const uchar* baseData = baseImage.constBits();
  const uchar* sourceData = sourceImage.constBits();
  const uchar* maskData = maskImage.constBits();
  uchar* resultData = result.bits();

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int offset = (y * width + x) * 4;

      // Extract the mask alpha value
      int maskAlpha = maskData[offset + 3];

      if(maskAlpha > alphaThreshold){
          //If the mask is opaque copy from the source
        resultData[offset] = sourceData[offset];
        resultData[offset + 1] = sourceData[offset + 1];
        resultData[offset + 2] = sourceData[offset + 2];
        resultData[offset + 3] = sourceData[offset + 3];
      } else{
          // otherwise copy from the base
        resultData[offset] = baseData[offset];
        resultData[offset + 1] = baseData[offset + 1];
        resultData[offset + 2] = baseData[offset + 2];
        resultData[offset + 3] = baseData[offset + 3];

      }
    }
  }
    return result;
}
```

In this example we have added a new image that will be used as a mask and a threshold value. The logic is now conditional, with different branches depending on the pixel's alpha value. If the mask’s alpha exceeds the provided threshold, then pixels are copied from the source; otherwise, pixels from the base image are copied into the result. This provides the capability to implement a variety of complex masking behaviours efficiently.

These examples demonstrate that `QImage` objects can be combined without relying on raster operations using direct pixel manipulation techniques. Performance gains can be substantial, however it is crucial to validate image formats and sizes, and be aware of potential issues related to memory alignment and access. Utilizing this methodology allowed my team to produce real-time image effects which were otherwise intractable.

For further study, I would recommend exploring resources that detail image processing algorithms. Understanding the basic principles of linear algebra can also help in understanding more advanced blending techniques. Investigating SIMD instruction sets relevant to the target platform will allow for additional optimization. Finally, the Qt documentation on `QImage` remains a crucial source for clarifying memory layout details and pixel formats.
