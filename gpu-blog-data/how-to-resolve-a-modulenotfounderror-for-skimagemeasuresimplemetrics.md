---
title: "How to resolve a `ModuleNotFoundError` for `skimage.measure.simple_metrics`?"
date: "2025-01-30"
id: "how-to-resolve-a-modulenotfounderror-for-skimagemeasuresimplemetrics"
---
The `ModuleNotFoundError` pertaining to `skimage.measure.simple_metrics` signals a misconfiguration or version incompatibility within the scikit-image library, a core component of many image processing pipelines. This specific module, present in earlier versions, was deprecated and removed in favor of a more streamlined approach to metric calculations. My experience migrating a large-scale image analysis project in 2021 highlighted this very issue; several legacy scripts relied directly on `simple_metrics`, which abruptly halted processing after a scikit-image update. The resolution hinges on understanding the underlying changes and adapting your code to utilize the current recommended functions.

The root cause is the removal of the `skimage.measure.simple_metrics` module, effectively splitting its functionality across other parts of the `skimage.measure` namespace. Commonly used metrics such as the mean squared error (MSE), peak signal-to-noise ratio (PSNR), and structural similarity index (SSIM) are now calculated using dedicated functions within `skimage.metrics` and `skimage.metrics.structural_similarity`. Therefore, attempting to import `skimage.measure.simple_metrics` after this change will inevitably result in the `ModuleNotFoundError` you are encountering. Resolving this necessitates a two-pronged approach: identifying where `simple_metrics` is used and replacing it with the equivalent contemporary functions.

The first step involves locating instances of the problematic import: `from skimage.measure import simple_metrics`. These import statements are the entry point for the error. Once found, we replace function calls to `simple_metrics` with calls to their appropriate replacement in `skimage.metrics`. Consider these code examples to illustrate the transition.

**Example 1: Calculating Mean Squared Error**

In older code, calculating the mean squared error using `simple_metrics` might look like this:

```python
# Old code (will raise ModuleNotFoundError with modern scikit-image)
import numpy as np
from skimage.measure import simple_metrics

image1 = np.random.rand(100, 100)
image2 = np.random.rand(100, 100)

mse_value = simple_metrics.mean_squared_error(image1, image2)
print(f"Old MSE: {mse_value}")
```
This code snippet uses a direct import of `simple_metrics` and calls its `mean_squared_error` function. It would not work with recent versions of `scikit-image`.  The fix entails removing this import and utilizing the `skimage.metrics.mean_squared_error` function:

```python
# Corrected code
import numpy as np
from skimage.metrics import mean_squared_error

image1 = np.random.rand(100, 100)
image2 = np.random.rand(100, 100)

mse_value = mean_squared_error(image1, image2)
print(f"New MSE: {mse_value}")
```

This demonstrates the direct substitution. The corrected code imports `mean_squared_error` directly from `skimage.metrics` and invokes it the same way, achieving equivalent functionality without triggering the `ModuleNotFoundError`.

**Example 2: Calculating Peak Signal-to-Noise Ratio (PSNR)**

Another common use case for `simple_metrics` was PSNR calculation. A similar transition needs to take place. Old code:
```python
# Old code (will raise ModuleNotFoundError with modern scikit-image)
import numpy as np
from skimage.measure import simple_metrics

image1 = np.random.rand(100, 100)
image2 = np.random.rand(100, 100)

psnr_value = simple_metrics.peak_signal_noise_ratio(image1, image2)
print(f"Old PSNR: {psnr_value}")
```
This pattern matches Example 1, utilizing `simple_metrics.peak_signal_noise_ratio`. To resolve the error,  we replace this call with the equivalent function `skimage.metrics.peak_signal_noise_ratio`:

```python
# Corrected code
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

image1 = np.random.rand(100, 100)
image2 = np.random.rand(100, 100)

psnr_value = peak_signal_noise_ratio(image1, image2)
print(f"New PSNR: {psnr_value}")
```
This revised snippet replaces the older import with a direct import of the function. Again, the logic and usage remain the same; only the location of the function has changed.

**Example 3: Calculating Structural Similarity Index (SSIM)**

The structural similarity index (SSIM) also moved. The old approach using  `simple_metrics` would have looked something like:
```python
# Old code (will raise ModuleNotFoundError with modern scikit-image)
import numpy as np
from skimage.measure import simple_metrics
from skimage.util import img_as_float

image1 = np.random.rand(100, 100)
image2 = np.random.rand(100, 100)

image1 = img_as_float(image1)
image2 = img_as_float(image2)


ssim_value = simple_metrics.structural_similarity(image1, image2)
print(f"Old SSIM: {ssim_value}")
```
The equivalent modern code using  `skimage.metrics.structural_similarity`:
```python
# Corrected code
import numpy as np
from skimage.metrics import structural_similarity
from skimage.util import img_as_float

image1 = np.random.rand(100, 100)
image2 = np.random.rand(100, 100)

image1 = img_as_float(image1)
image2 = img_as_float(image2)

ssim_value = structural_similarity(image1, image2)
print(f"New SSIM: {ssim_value}")
```

Notice the consistent pattern:  we replace `from skimage.measure import simple_metrics` and then substitute any call to `simple_metrics.some_metric` with the equivalent in `skimage.metrics`.  Crucially, the input parameters and return values remain compatible between the deprecated and current versions for these metrics.

To further solidify understanding and best practices, consult several resources. The scikit-image API documentation, specifically the `skimage.metrics` module section, offers comprehensive details on the new function locations and any slight differences in behavior (though minimal for the metrics discussed).  The scikit-image example gallery provides practical examples for metric usage, which helps illustrate real world application beyond minimal code snippets. Finally, the scikit-image release notes are valuable; they explain in detail changes from earlier versions, including the deprecation of `simple_metrics` and the rationale for the restructuring. Examining release notes prior to library updates will highlight potential breaking changes in advance.

In conclusion, resolving the `ModuleNotFoundError` for `skimage.measure.simple_metrics` involves replacing the outdated module import and function calls with their contemporary equivalents residing in `skimage.metrics`. This requires a systematic search, replacement, and verification process. I recommend thorough testing after any such modification to guarantee accuracy, particularly when working with production-grade image processing pipelines. The example transformations provided represent the primary method of correcting older scripts, with the resources mentioned supplementing your efforts to ensure a smooth transition.
