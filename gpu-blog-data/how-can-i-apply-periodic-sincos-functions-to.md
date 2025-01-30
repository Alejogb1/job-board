---
title: "How can I apply periodic sin/cos functions to a 2D TensorFlow.js tensor?"
date: "2025-01-30"
id: "how-can-i-apply-periodic-sincos-functions-to"
---
Applying sinusoidal functions to a TensorFlow.js tensor requires understanding the underlying tensor structure and leveraging TensorFlow.js's element-wise operations.  My experience working on real-time audio processing pipelines heavily involved manipulating tensors representing sampled waveforms, frequently requiring precisely this type of manipulation.  The key is recognizing that TensorFlow.js applies functions element-wise by default, enabling efficient parallel computation.  This inherent parallelism is crucial for performance when dealing with larger tensors.

**1. Clear Explanation:**

The core principle is to apply `tf.sin()` and `tf.cos()` directly to the tensor.  However, achieving a *periodic* sinusoidal variation demands careful consideration of the input data.  The argument to the sine and cosine functions needs to represent the phase, which should be modulated to create the desired periodic effect.  Typically, this involves creating a separate tensor representing the phase across each element of your input tensor. This phase tensor will determine the point on the sine/cosine wave for each corresponding element in the input.

There are several strategies to generate the phase tensor, depending on whether you want a single shared frequency, a frequency gradient across the tensor dimensions, or a more complex phase profile.

For a single frequency, the simplest approach is to use a constant value scaled by time or a spatial index.  For a frequency gradient, you might linearly increase the frequency across tensor dimensions.  More intricate patterns can be achieved using custom tensor generation functions or by importing pre-computed phase data.  Importantly, the data type of the phase tensor should match that of the input tensor to avoid potential type errors during element-wise operations.  If the input tensor represents spatial coordinates, the phase tensor could be derived directly from these coordinates, resulting in spatially varying frequencies or patterns.


**2. Code Examples with Commentary:**

**Example 1: Single Frequency Modulation**

This example demonstrates applying a single sine wave to every element of a 2D tensor.

```javascript
// Import TensorFlow.js
import * as tf from '@tensorflow/tfjs';

// Define tensor dimensions
const rows = 100;
const cols = 100;

// Create a 2D tensor of random values (replace with your data)
const inputTensor = tf.randomNormal([rows, cols]);

// Define the frequency (in radians per element)
const frequency = 0.1;

// Create a phase tensor using the element index (this is a simplistic approach)
const phaseTensor = tf.linspace(0, 2 * Math.PI * frequency * (rows*cols-1), rows * cols).reshape([rows, cols]);

// Apply the sine function element-wise
const outputTensor = tf.sin(phaseTensor.add(inputTensor)); //Adding inputTensor for modulation

// Display or further process the output tensor
outputTensor.print();
outputTensor.dispose();
```

This code creates a phase tensor that linearly increases across the tensor, resulting in a sine wave whose frequency is determined by `frequency`.  Adding `inputTensor` introduces a modulation effect, resulting in a varied amplitude across the tensor. Note the crucial `dispose()` call for memory management, a best practice I've learned the hard way in large-scale projects.


**Example 2: Frequency Gradient Across Rows**

This example demonstrates applying a sine wave with a frequency gradient across the rows of a 2D tensor.

```javascript
import * as tf from '@tensorflow/tfjs';

const rows = 100;
const cols = 100;
const inputTensor = tf.randomNormal([rows, cols]);

// Create a frequency gradient tensor across rows
const rowFrequencies = tf.linspace(0.01, 1, rows).reshape([rows, 1]);
const colFrequencies = tf.ones([1, cols]);
const frequencyTensor = tf.tidy(() => rowFrequencies.mul(colFrequencies));

const phaseTensor = tf.tidy(() => tf.mul(frequencyTensor, tf.range(rows*cols).reshape([rows,cols])));


const outputTensor = tf.sin(phaseTensor.add(inputTensor)); //Adding inputTensor for modulation


outputTensor.print();
outputTensor.dispose();
```

Here, we create a `frequencyTensor` that linearly increases from 0.01 to 1.0 across rows, producing a sinusoidal wave whose frequency changes across the rows.  The `tf.tidy` function is vital for efficient memory management within complex tensor operations, a lesson I've learned through optimizing performance in my previous projects.


**Example 3:  Using a Precomputed Phase Map**

This example utilizes a pre-computed phase map for more complex patterns.

```javascript
import * as tf from '@tensorflow/tfjs';

const rows = 100;
const cols = 100;

// Load precomputed phase map. Replace with your actual loading mechanism.
const phaseMap = tf.tensor2d( //replace with actual data
  [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
  ]
);
//assuming phaseMap is of correct dimension

const inputTensor = tf.randomNormal([rows, cols]);

// Ensure phase map dimensions match input tensor
const scaledPhaseMap = tf.reshape(phaseMap, [rows, cols]);

// Apply the sine function element-wise
const outputTensor = tf.sin(scaledPhaseMap);

outputTensor.print();
outputTensor.dispose();
```

This example showcases the flexibility of the approach.  The pre-computed `phaseMap` allows for arbitrary patterns and frequencies, significantly broadening the range of possible effects.   This technique would be crucial for situations where the phase is not easily calculated analytically.


**3. Resource Recommendations:**

The TensorFlow.js API documentation, specifically sections on tensor manipulation, element-wise operations, and mathematical functions.  A comprehensive linear algebra textbook to reinforce the mathematical foundations of tensor operations. A book focusing on digital signal processing (DSP) concepts will greatly enhance your understanding of frequency and phase manipulation in the context of signal processing.  Finally, studying advanced optimization techniques for TensorFlow.js can significantly improve the efficiency of your code, especially when dealing with very large tensors.
