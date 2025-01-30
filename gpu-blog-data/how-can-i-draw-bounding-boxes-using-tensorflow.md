---
title: "How can I draw bounding boxes using TensorFlow in C++?"
date: "2025-01-30"
id: "how-can-i-draw-bounding-boxes-using-tensorflow"
---
Bounding box drawing in TensorFlow C++ requires careful management of tensor data, image manipulation, and potentially graphics library integration. Accessing model outputs, often in the form of bounding box coordinates and class predictions, necessitates unpacking these tensors and converting them to a usable format for drawing rectangles onto an image. My experience working on real-time object detection systems has highlighted the common challenges and efficient strategies in this process.

The core steps involve obtaining prediction output from a TensorFlow model, parsing these outputs into interpretable bounding box coordinates, and then utilizing an appropriate drawing method. The typical scenario involves a model producing a tensor representing bounding box locations (e.g., coordinates `ymin`, `xmin`, `ymax`, `xmax`), and another tensor containing the class labels (or scores) associated with each bounding box. These tensors are usually floating-point values, normalized to the range [0, 1], with respect to the image dimensions. Therefore, conversion to pixel coordinates is usually the first necessary step.

**1. Parsing TensorFlow Output Tensors**

The initial hurdle is accessing the relevant data from the output tensors returned by the TensorFlow session. After running inference, you’ll typically receive one or more output tensors as `tensorflow::Tensor` objects. The key lies in understanding the shape and data type of these tensors. For bounding boxes, the shape is usually `[N, 4]`, where `N` is the number of detected boxes, and the four values represent the normalized `ymin`, `xmin`, `ymax`, and `xmax` coordinates, respectively. Class label tensors are usually of shape `[N, 1]` or `[N, C]` where `C` represents the number of classes, and a maximum score of the `C` is considered to represent the class.

```c++
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/tensor.h>
#include <vector>
#include <iostream>

struct BoundingBox {
  float ymin, xmin, ymax, xmax;
  int classId;
  float score;
};

std::vector<BoundingBox> parseBoundingBoxes(const tensorflow::Tensor& boxesTensor,
                                            const tensorflow::Tensor& classesTensor,
                                            const tensorflow::Tensor& scoresTensor,
                                            int imageWidth,
                                            int imageHeight,
                                            float scoreThreshold)
{
  std::vector<BoundingBox> boundingBoxes;
  auto boxes = boxesTensor.tensor<float, 2>();
  auto classes = classesTensor.tensor<float, 2>();
  auto scores = scoresTensor.tensor<float, 2>();

  int numBoxes = boxesTensor.dim_size(0);
  for (int i = 0; i < numBoxes; ++i) {
    float score = scores(i, 0);
    if (score >= scoreThreshold) {
        float ymin = boxes(i, 0) * imageHeight;
        float xmin = boxes(i, 1) * imageWidth;
        float ymax = boxes(i, 2) * imageHeight;
        float xmax = boxes(i, 3) * imageWidth;
        int classId = static_cast<int>(classes(i, 0));

        boundingBoxes.push_back({ymin, xmin, ymax, xmax, classId, score});
    }
  }
  return boundingBoxes;
}

//Example of how to call the function
/*
  tensorflow::Tensor boxesTensor; // Filled with output data
  tensorflow::Tensor classesTensor; // Filled with output data
  tensorflow::Tensor scoresTensor; // Filled with output data

  int imageWidth = 640;
  int imageHeight = 480;
  float scoreThreshold = 0.5;

  std::vector<BoundingBox> detectedBoxes = parseBoundingBoxes(boxesTensor, classesTensor, scoresTensor, imageWidth, imageHeight, scoreThreshold);
  
  for (const auto& box : detectedBoxes) {
      std::cout << "Box: ymin=" << box.ymin << ", xmin=" << box.xmin 
                << ", ymax=" << box.ymax << ", xmax=" << box.xmax 
                << ", classId=" << box.classId << ", score=" << box.score << std::endl;
  }
*/
```

This code illustrates how to unpack bounding box data, class IDs, and scores from TensorFlow output tensors. It iterates through each detected box, converts the normalized coordinates to pixel coordinates using the image dimensions, applies a score threshold, and stores only boxes above the score threshold in the `boundingBoxes` vector. Using the `tensor<float, 2>()` method provides direct access to tensor data for efficiency, avoiding extra copies. The critical point is to match the `tensor<>()` method's template parameters with the data type and shape of the output tensor.

**2. Image Manipulation and Drawing**

Once bounding box coordinates are obtained, they must be applied to the image. The specific method depends on the representation of your input image and available drawing libraries. If you are working with image data loaded into a `tensorflow::Tensor`, directly modifying the tensor might not be the most performant approach, especially when dealing with common pixel formats like RGB, since tensorflow tensors do not directly support drawing. More practical solutions would involve loading the image via OpenCV or other libraries that provide such functionality to draw directly on images.

```c++
#include <opencv2/opencv.hpp>
#include <vector>

struct BoundingBox {
  float ymin, xmin, ymax, xmax;
  int classId;
  float score;
};


void drawBoundingBoxes(cv::Mat& image, const std::vector<BoundingBox>& boundingBoxes) {
  for (const auto& box : boundingBoxes) {
      cv::Point topLeft(static_cast<int>(box.xmin), static_cast<int>(box.ymin));
      cv::Point bottomRight(static_cast<int>(box.xmax), static_cast<int>(box.ymax));
      cv::Scalar color(0, 255, 0); // Green color for bounding boxes
      cv::rectangle(image, topLeft, bottomRight, color, 2);

      std::string label = "Class " + std::to_string(box.classId) + " (" + std::to_string(box.score) + ")";
      int baseline = 0;
      cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::Point textOrg(static_cast<int>(box.xmin), static_cast<int>(box.ymin) - 5);
      cv::rectangle(image, textOrg + cv::Point(0, textSize.height) , textOrg + cv::Point(textSize.width, -baseline), color, cv::FILLED);
      cv::putText(image, label, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

// Example usage
/*
  cv::Mat image = cv::imread("image.jpg");
  std::vector<BoundingBox> boxes = ... // Your detected bounding box data
  drawBoundingBoxes(image, boxes);
  cv::imshow("Bounding Boxes", image);
  cv::waitKey(0);
*/
```

This example leverages OpenCV for image loading and drawing. It iterates through each `BoundingBox` and converts the float coordinates to integer pixel coordinates, which OpenCV `rectangle` requires. It draws the bounding box rectangle with a fixed color and also writes the class label and score on top of each bounding box. The core idea is to bridge the data produced by TensorFlow (the numerical coordinates) with a practical way to render them on an image using an available library.

**3. Generalization with a Generic `draw` function**
An ideal solution would encapsulate the entire detection drawing procedure with a single generic function, given that most libraries provide similar capabilities. This example shows a simplified function, with less specific image library calls:

```c++
#include <vector>
#include <string>

struct BoundingBox {
  float ymin, xmin, ymax, xmax;
  int classId;
  float score;
};

//Interface for the drawing library
class DrawingContext {
public:
    virtual void drawRectangle(int x1, int y1, int x2, int y2, int color) = 0;
    virtual void drawText(int x, int y, const std::string& text, int color) = 0;
    virtual int getTextHeight(const std::string& text) = 0;
    virtual int getTextWidth(const std::string& text) = 0;
    virtual ~DrawingContext() = default;
};
//Example of a context implementation
class ConsoleDrawingContext : public DrawingContext{
public:
  void drawRectangle(int x1, int y1, int x2, int y2, int color) override {
    std::cout << "Drawing Rectangle: x1=" << x1 << ", y1=" << y1 << ", x2=" << x2 << ", y2=" << y2 << ", color=" << color << std::endl;
  };
  void drawText(int x, int y, const std::string& text, int color) override {
    std::cout << "Drawing text: x=" << x << ", y=" << y << ", text=" << text << ", color=" << color << std::endl;
  };
    int getTextHeight(const std::string& text) override{
    return 10;
  }
    int getTextWidth(const std::string& text) override{
    return 2 * text.size();
  }
};

void drawBoundingBoxesGeneric(DrawingContext& context, const std::vector<BoundingBox>& boundingBoxes) {
  for (const auto& box : boundingBoxes) {
    int x1 = static_cast<int>(box.xmin);
    int y1 = static_cast<int>(box.ymin);
    int x2 = static_cast<int>(box.xmax);
    int y2 = static_cast<int>(box.ymax);
    int color = 0x00FF00; // Green color in RGB

    context.drawRectangle(x1, y1, x2, y2, color);

    std::string label = "Class " + std::to_string(box.classId) + " (" + std::to_string(box.score) + ")";
    int textHeight = context.getTextHeight(label);
    int textWidth = context.getTextWidth(label);
    
    context.drawRectangle(x1, y1 - textHeight, x1 + textWidth, y1, color);
    context.drawText(x1, y1 - 2, label, 0x000000); //Black color for the text
  }
}

//Example usage:
/*
  std::vector<BoundingBox> boxes = ... // Your detected bounding box data
  ConsoleDrawingContext context;
  drawBoundingBoxesGeneric(context, boxes);
*/
```

This revised design decouples the core logic from any specific drawing library, enabling an abstract interface based approach. The `DrawingContext` class provides an interface that can be implemented to support specific libraries such as OpenCV or even a software rendering library. This allows you to swap the drawing mechanism without modifying the main bounding box drawing logic. This example implements a simple `ConsoleDrawingContext` to illustrate how you can draw using the terminal console. This approach will facilitate modularity and extensibility.

**Resource Recommendations:**

To deepen your understanding, I recommend reviewing documentation for the TensorFlow C++ API, especially focusing on the `tensorflow::Session`, `tensorflow::Tensor`, and their respective classes. Examine examples involving model loading and inference. Tutorials and examples from the OpenCV library will be vital for understanding image loading, basic image transformations, and fundamental drawing operations. I have also found that exploring example implementations from GitHub repositories or similar platforms can clarify how others have implemented these processes in practice. Finally, understand the specific tensor output formats of the model being used by carefully examining the model documentation.
