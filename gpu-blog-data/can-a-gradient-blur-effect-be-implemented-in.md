---
title: "Can a gradient blur effect be implemented in a Flutter custom paint?"
date: "2025-01-30"
id: "can-a-gradient-blur-effect-be-implemented-in"
---
Implementing a gradient blur effect within Flutter's custom paint functionality is indeed possible, although it requires understanding how Flutter's canvas and Skia interact. The core challenge lies in the fact that Skia, the graphics library Flutter uses, doesn't provide a single operation that directly achieves a "gradient blur." We must create this effect through a combination of layered painting and careful use of blur filters.

The standard `Canvas` drawing operations focus on directly rendering shapes, paths, and text onto the canvas. While `Canvas.drawRect()`, `Canvas.drawPath()`, and other methods facilitate precise drawing, they lack inherent blur functionality linked to a gradient. Instead, we must approach it through drawing a gradient, and then blurring this painted area. My experience developing custom visualizations for a financial charting application highlighted this directly; complex blending and masking strategies were necessary for nuanced effects.

The first step is to draw the gradient we want to blur. Flutter's `Paint` class offers a `shader` property. We can set this to a `Gradient`, which can be of various types, such as `LinearGradient`, `RadialGradient`, or `SweepGradient`. This creates a color transition across the defined area. We then use `canvas.drawRect` to paint a rectangular region with this shader.

Following this, we don't directly "blur the gradient" on the canvas. Instead, we treat the painted area like a bitmap, then apply a blur. To accomplish this, we must first draw to an offscreen canvas. Specifically, we use `ui.PictureRecorder` to capture a picture of the drawn gradient rect and then we get a `ui.Image` from it. The next stage applies the blur using `ImageFilter`. This filter takes a sigma (blur radius) as input and can be applied to the created image using `canvas.drawImageRect` by drawing from source `image` to the destination rect. Critically, the blurry destination rect can optionally be larger than the source rect to achieve an overall blurred effect in which the colors appear to run beyond the original bounds.

The final step is to draw this blurred image onto our main canvas. This composite process gives the impression of a blurred gradient. This approach isn't the most computationally inexpensive, as it involves offscreen rendering and image manipulation, so careful management of resource is important, especially for high refresh rates.

Here's a code example illustrating the process:

```dart
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class GradientBlurPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final gradientRect = Rect.fromLTWH(0, 0, size.width, size.height);

    final gradientPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Colors.blue, Colors.green],
      ).createShader(gradientRect);

     final recorder = ui.PictureRecorder();
     final recorderCanvas = Canvas(recorder);
     recorderCanvas.drawRect(gradientRect, gradientPaint);
     final picture = recorder.endRecording();
     final image = picture.toImageSync(size.width.toInt(), size.height.toInt());

     final sigma = 10.0;
     final blurPaint = Paint()
      ..imageFilter = ui.ImageFilter.blur(sigmaX: sigma, sigmaY: sigma);

      final blurredRect = Rect.fromLTWH(-sigma, -sigma, size.width + 2 * sigma, size.height + 2 * sigma);
    
    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, size.width.toDouble(), size.height.toDouble()),
      blurredRect,
      blurPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class GradientBlurWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: GradientBlurPainter(),
      size: Size(200, 100), // Define the size of the painting area.
    );
  }
}
```

In this first example, a linear gradient from blue to green is created within the `GradientBlurPainter`. A `PictureRecorder` and `Canvas` are used to draw to a new image. A `blur` filter using `ImageFilter` is then applied, resulting in a blur of 10 pixels in both the X and Y directions. The `sigma` value controls the strength of the blur. The blurred image is drawn to canvas with a source rectangle, and a blurred destination rect which is larger than the source. The `GradientBlurWidget` uses this painter to paint on the screen.

The next example demonstrates a radial gradient:

```dart
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class RadialGradientBlurPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
      final gradientRect = Rect.fromLTWH(0, 0, size.width, size.height);

    final gradientPaint = Paint()
      ..shader = RadialGradient(
        center: Alignment.center,
        radius: 0.5,
        colors: [Colors.red, Colors.yellow],
      ).createShader(gradientRect);

     final recorder = ui.PictureRecorder();
     final recorderCanvas = Canvas(recorder);
     recorderCanvas.drawRect(gradientRect, gradientPaint);
     final picture = recorder.endRecording();
     final image = picture.toImageSync(size.width.toInt(), size.height.toInt());

     final sigma = 15.0;
     final blurPaint = Paint()
      ..imageFilter = ui.ImageFilter.blur(sigmaX: sigma, sigmaY: sigma);
    
       final blurredRect = Rect.fromLTWH(-sigma, -sigma, size.width + 2 * sigma, size.height + 2 * sigma);

    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, size.width.toDouble(), size.height.toDouble()),
      blurredRect,
      blurPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class RadialGradientBlurWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: RadialGradientBlurPainter(),
      size: Size(200, 100),
    );
  }
}
```

The change here is the `LinearGradient` replaced with `RadialGradient` with a center alignment, showing that we can use different gradients to achieve a similar blur effect. The blur is increased to 15 pixels, which demonstrates how the radius effects the output blur.

Finally, let's consider an example that incorporates different areas for the gradient and blurred images:

```dart
import 'dart:ui' as ui;
import 'package:flutter/material.dart';

class CustomAreaGradientBlurPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final gradientRect = Rect.fromLTWH(size.width / 4, size.height / 4, size.width / 2, size.height / 2);

    final gradientPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Colors.purple, Colors.pink],
      ).createShader(gradientRect);

    final recorder = ui.PictureRecorder();
    final recorderCanvas = Canvas(recorder);
    recorderCanvas.drawRect(gradientRect, gradientPaint);
    final picture = recorder.endRecording();
    final image = picture.toImageSync(size.width.toInt(), size.height.toInt());

    final sigma = 8.0;
    final blurPaint = Paint()
      ..imageFilter = ui.ImageFilter.blur(sigmaX: sigma, sigmaY: sigma);

      final blurredRect = Rect.fromLTWH(0, 0, size.width, size.height);

    canvas.drawImageRect(
      image,
      Rect.fromLTWH(0, 0, size.width.toDouble(), size.height.toDouble()),
      blurredRect,
      blurPaint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class CustomAreaGradientBlurWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: CustomAreaGradientBlurPainter(),
      size: Size(200, 200),
    );
  }
}
```

This final example creates a gradient within a rect which is smaller and off-center, at coordinates `(size.width / 4, size.height / 4)`, with a width and height of half the size of the overall available area. This demonstrates how a gradient can be drawn and blurred within specific areas of a larger canvas. The blur is applied to the entire surface area of the canvas.

For further exploration, I recommend focusing on the following areas:

1.  **Skia Documentation**: Understanding the underlying Skia library is vital to optimizing custom painting. I found that studying their API documentation significantly improved my handling of canvas operations and resource management.
2.  **Flutter's CustomPaint Documentation**: Thoroughly review Flutter’s documentation on `CustomPaint`, `Canvas`, and `Paint` classes. This provides a comprehensive understanding of available drawing and painting tools. The Flutter cookbook also contains numerous examples of custom painters and the use of the `Canvas` class.
3.  **Performance Considerations**: Explore performance optimization techniques, including limiting offscreen operations and considering caching of images for static content within `CustomPainter` implementations, to ensure smooth animations at 60 fps.

Implementing gradient blurs in Flutter’s custom paint is a multi-stage process, and the examples demonstrate the key components. The complexity comes from the fact that this is not a direct canvas drawing operation. Instead, we must paint an image, then apply an `ImageFilter` to it, and then redraw it to the canvas. This approach provides great flexibility but requires careful design and resource management.
