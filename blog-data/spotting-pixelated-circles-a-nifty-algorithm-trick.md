---
title: "Spotting Pixelated Circles: A Nifty Algorithm Trick"
date: '2024-11-08'
id: 'spotting-pixelated-circles-a-nifty-algorithm-trick'
---

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
  Mat image = imread("EuDpD.png", IMREAD_GRAYSCALE);

  // Kernel for convolution
  float kernel1Data[] = {
    -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,
    -9, -7, -2, -1, 0, 0, 0, 0, -1, -2, -7, -9,
    -9, -2, -1, 0, 9, 9, 9, 9, 0, -1, -2, -9,
    -9, -1, 0, 9, 7, 7, 7, 7, 9, 0, -1, -9,
    -9, 0, 9, 7, -9, -9, -9, -9, 7, 9, 0, -9,
    -9, 0, 9, 7, -9, -9, -9, -9, 7, 9, 0, -9,
    -9, 0, 9, 7, -9, -9, -9, -9, 7, 9, 0, -9,
    -9, 0, 9, 7, -9, -9, -9, -9, 7, 9, 0, -9,
    -9, -1, 0, 9, 7, 7, 7, 7, 9, 0, -1, -9,
    -9, -2, 0, 0, 9, 9, 9, 9, 0, 0, -2, -9,
    -9, -7, -2, -1, 0, 0, 0, 0, -1, -2, -7, -9,
    -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9
  };
  Mat kernel1 = Mat(12, 12, CV_32F, kernel1Data) / 240;

  // Kernel for sharpening
  float sharpenData[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
  Mat sharpen = Mat(3, 3, CV_32F, sharpenData);

  // Apply convolution
  Mat identify;
  filter2D(image, identify, -1, kernel1);

  // Sharpen image
  filter2D(identify, identify, -1, sharpen);

  // Thresholding
  Mat thresh;
  threshold(identify, thresh, 220, 255, THRESH_BINARY);

  imwrite("identify.png", thresh);

  return 0;
}
```
