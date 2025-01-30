---
title: "How can I feed an OpenCV Mat to a TensorFlow C++ graph?"
date: "2025-01-30"
id: "how-can-i-feed-an-opencv-mat-to"
---
Transferring image data between OpenCV's `cv::Mat` and TensorFlow's C++ API requires careful memory management and data type conversion, as these libraries do not natively share a common data representation. I've faced this challenge numerous times while integrating real-time image processing pipelines with machine learning models. The core of the problem lies in bridging the gap between OpenCV's matrix-like structure and TensorFlow's tensor structure, primarily involving memory layout and data precision.

At its foundation, a `cv::Mat` stores image data in a variety of formats (e.g., `CV_8UC3` for 8-bit unsigned chars with three channels representing a color image), while TensorFlow's C++ API expects a `tensorflow::Tensor`, which is a multidimensional array. The memory layout of `cv::Mat` is typically row-major, whereas TensorFlow’s internal handling of tensors is more flexible, often requiring an explicit reshaping or permutation to align with the graph's input requirements. Furthermore, data types might differ. For instance, images are often read as 8-bit integers by OpenCV, but TensorFlow models frequently expect floating-point representations between 0.0 and 1.0. Consequently, the most efficient path involves copying data with an appropriate scaling and type conversion.

Let's consider a common scenario: feeding a color image obtained using OpenCV into a TensorFlow graph that expects a 4D tensor of shape `[1, height, width, channels]` with `float` precision.

**Code Example 1: Basic Data Transfer and Conversion**

```c++
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>

tensorflow::Tensor matToTensor(const cv::Mat& image) {
  // 1. Pre-allocate memory for TensorFlow tensor
  int height = image.rows;
  int width = image.cols;
  int channels = image.channels();
  tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, channels}));

  // 2. Get raw memory pointer to the tensor
  auto tensorMap = inputTensor.flat<float>().data();

  // 3. Convert image data to float, scaling to range [0, 1]
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
        for(int k = 0; k < channels; k++) {
            tensorMap[(i * width + j) * channels + k] =
                static_cast<float>(image.at<cv::Vec3b>(i, j)[k]) / 255.0f;
        }
    }
  }

  return inputTensor;
}

// Example Usage:
// ... (Loading TensorFlow model, setting up session etc. ) ...
cv::Mat image = cv::imread("test.jpg");
if (image.empty()) {
    std::cerr << "Error: Could not load image." << std::endl;
    return 1;
}
cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Ensure the right color format

tensorflow::Tensor tensor_input = matToTensor(image);
// ... Use inputTensor to feed to TensorFlow session
```

In this first example, `matToTensor` function pre-allocates memory for the TensorFlow tensor with the specified shape and data type using `tensorflow::TensorShape`. The key point is accessing the underlying raw memory pointer using `inputTensor.flat<float>().data()`. This pointer is crucial for efficient data transfer. The nested loops then iterate through each pixel of the `cv::Mat` image and converts the 8-bit unsigned char values to floating-point data, simultaneously scaling them to the range of [0.0, 1.0] by dividing by 255.0f. This is critical because neural network models frequently operate with normalized floating-point data.

**Code Example 2: Using `memcpy` for Improved Performance**

```c++
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>

tensorflow::Tensor matToTensorOptimized(const cv::Mat& image) {
  // 1. Pre-allocate memory for TensorFlow tensor
  int height = image.rows;
  int width = image.cols;
  int channels = image.channels();
  tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, channels}));

  // 2. Get raw memory pointer to the tensor
  float* tensorMap = inputTensor.flat<float>().data();

  // 3. Convert image data to float, scaling to range [0, 1] using memcpy
  const unsigned char* imageData = image.data;
  size_t totalSize = height * width * channels;
    
    for (size_t i = 0; i < totalSize; i++) {
       tensorMap[i] = static_cast<float>(imageData[i])/255.0f;
    }

  return inputTensor;
}

// Example Usage:
// ... (Loading TensorFlow model, setting up session etc. ) ...
cv::Mat image = cv::imread("test.jpg");
if (image.empty()) {
    std::cerr << "Error: Could not load image." << std::endl;
    return 1;
}
cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Ensure the right color format

tensorflow::Tensor tensor_input = matToTensorOptimized(image);
// ... Use inputTensor to feed to TensorFlow session
```

The second example modifies the previous one to use a more efficient single loop and uses the raw `cv::Mat` pointer for data retrieval. It does not use the `image.at<cv::Vec3b>` function, which incurs some overhead by checking bounds during each pixel access. Instead, `image.data` is used, accessing a contiguous memory region, significantly improving speed, especially for large images. The crucial optimization is now in using a single loop that performs a direct read, conversion, and scaling into the correct location of the TensorFlow tensor.

**Code Example 3: Leveraging OpenCV's `convertTo` Method**

```c++
#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>

tensorflow::Tensor matToTensorOpenCV(const cv::Mat& image) {
  // 1. Convert to floating point, scaled to [0, 1] using convertTo
  cv::Mat floatImage;
  image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

  // 2. Create TensorFlow tensor
  int height = floatImage.rows;
  int width = floatImage.cols;
  int channels = floatImage.channels();
  tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height, width, channels}));

  // 3. Copy data into tensor
  float* tensorData = inputTensor.flat<float>().data();
  std::memcpy(tensorData, floatImage.data, floatImage.total() * floatImage.elemSize());
  
  return inputTensor;
}


// Example Usage:
// ... (Loading TensorFlow model, setting up session etc. ) ...
cv::Mat image = cv::imread("test.jpg");
if (image.empty()) {
    std::cerr << "Error: Could not load image." << std::endl;
    return 1;
}
cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // Ensure the right color format

tensorflow::Tensor tensor_input = matToTensorOpenCV(image);
// ... Use inputTensor to feed to TensorFlow session
```

The final example utilizes OpenCV’s built-in `convertTo` method, which provides a highly optimized mechanism for performing scaling and type conversions. I found this to be generally the fastest option, especially when combined with `memcpy` for the final transfer to the tensor’s memory. First, it converts the original image to a floating-point image using `CV_32FC3` as a target, incorporating the scaling to [0, 1] within the conversion itself. Afterwards, a `tensorflow::Tensor` is allocated, and its raw memory pointer is retrieved. Finally, `memcpy` is used to transfer the data from the converted float image directly to the tensor’s data buffer. This approach is typically the most computationally efficient because `convertTo` is highly optimized for various data type changes and avoids explicit manual looping over pixels.

In summary, feeding an OpenCV `cv::Mat` to TensorFlow requires carefully managing data conversion and memory handling. While manual iteration and casting are viable, they are generally slower than leveraging optimized functions. For my use cases, the last method using `convertTo` with `memcpy` consistently provided the best performance.

For further study, I recommend consulting the following resources: *OpenCV documentation* on the `cv::Mat` class, especially focusing on data access methods and conversions, and the *TensorFlow C++ API documentation* on creating tensors and memory management. Studying *efficient memory handling and buffer copying techniques in C++* is valuable, as is gaining an understanding of *data type conversion principles*. Examining examples within the *TensorFlow source code* concerning image processing might provide additional insights. Understanding how data is laid out in memory for both libraries is vital for avoiding errors and optimizing performance.
