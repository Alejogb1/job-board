---
title: "How can I integrate a Keras model (converted from .h5 to .json) into an HTML web page?"
date: "2025-01-30"
id: "how-can-i-integrate-a-keras-model-converted"
---
The challenge in deploying a Keras model on a client-side web page arises from the fundamental difference in execution environments: Python-based model training versus JavaScript-based web browsers. A direct port of a Keras model from its native format (.h5) or a JSON configuration file is not executable in standard browsers. Instead, we leverage TensorFlow.js, a JavaScript library, as a bridge. My experience building a live hand gesture recognition application several years ago underscores this point. I initially attempted to implement a Python Flask backend, processing images sent from the browser, but latency made the real-time performance unsatisfactory. I subsequently switched to a client-side implementation using TensorFlow.js, which substantially improved responsiveness. This experience taught me the complexities involved in moving machine learning models to web-based environments.

To integrate a Keras model into an HTML page, several steps are necessary: model conversion, web page setup, and JavaScript implementation using TensorFlow.js. Converting an .h5 file to .json is only the first step; we need the model's weight data as well. These two components, the model structure and the weights, will be loaded into a TensorFlow.js model.

First, convert the Keras .h5 model to a TensorFlow.js compatible format. This conversion involves generating two output files: a JSON file containing the model architecture, and a binary file, often in `.bin` or `.weights.bin` formats, containing the weights. While it's technically possible to load directly from an h5, the recommended approach involves converting to the TensorFlow.js format, as it is optimized for this specific usage. The Python script for this conversion utilizes `tensorflowjs` library, installed via pip using `pip install tensorflowjs`.

```python
import tensorflow as tf
import tensorflowjs as tfjs

# Load the Keras model
model = tf.keras.models.load_model('my_model.h5')

# Convert the model to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'tfjs_model')
```

This script generates a 'tfjs_model' folder containing `model.json`, which describes the structure, and a group of `weights.bin` files, which hold the learned parameters. The next step involves creating the HTML structure necessary for interaction. A basic setup requires at least an HTML file, and ideally a separate JavaScript file to handle the TensorFlow.js logic.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Keras Model on Web</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <script src="script.js" defer></script>
</head>
<body>
    <div id="output">Model output will appear here.</div>
    <input type="file" id="imageUpload" accept="image/*">
    <img id="preview" style="max-width:300px; max-height:300px;">
</body>
</html>
```

This HTML includes a reference to the TensorFlow.js library from a CDN, as well as an external JavaScript file named `script.js`. It contains a `div` to display model output, an image upload input, and an image preview element. The actual logic of model loading, image preprocessing, and prediction is handled within the `script.js` file.

```javascript
document.addEventListener('DOMContentLoaded', async () => {

    const model = await tf.loadLayersModel('tfjs_model/model.json');
    const imageUpload = document.getElementById('imageUpload');
    const outputDiv = document.getElementById('output');
    const previewImage = document.getElementById('preview');

    imageUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        previewImage.src = URL.createObjectURL(file);

        const image = new Image();
        image.src = URL.createObjectURL(file);

        image.onload = async () => {
            const tensor = tf.browser.fromPixels(image)
                              .resizeNearestNeighbor([64, 64]) // Preprocessing: Resize as per my model
                              .toFloat()
                              .div(tf.scalar(255.0))
                              .expandDims();

            const prediction = model.predict(tensor);
            const label = await prediction.argMax(1).data(); // Get the predicted class index

            outputDiv.innerText = `Predicted class: ${label[0]}`;

            tensor.dispose();
            prediction.dispose();
        };

    });
});
```

This `script.js` first loads the model upon page load. It listens for a file upload event. On upload, it reads the selected image file, converts it to a tensor using `tf.browser.fromPixels`, resizes it according to the input dimensions of my model (64x64 in this example), converts it to a floating-point representation, normalizes it by dividing by 255, and expands the dimensions to prepare for a batch input. The model is then run via the `model.predict()` function. The predicted output is then converted to a readable index, which will serve as the class prediction and subsequently displayed to the user. Finally, the tensors are disposed of to manage memory effectively. The use of `async`/`await` is important to properly handle asynchronous operations, such as loading the model or processing images. The resize, float, and division by 255 are examples of preprocessing steps that are dependent on the way the original Keras model was trained. My experience with image recognition highlighted how crucial it is to meticulously replicate the preprocessing from the training stage to obtain accurate predictions. It is essential for the developer to know the preprocessing steps used to train the model in python and perform equivalent steps in JavaScript using TensorFlow.js

For resources, the official TensorFlow.js documentation is crucial for understanding the API and functionalities. Additionally, examples available on the TensorFlow website provide practical insights into model loading and inference. Furthermore, GitHub repositories containing TensorFlow.js demos offer a broad understanding of integration approaches and best practices. Detailed blog posts from individuals who have shared similar experiences, often provide real-world examples. Finally, the TensorFlow.js GitHub issues page can be a place to find answers to complex and common questions, and a way to stay abreast of current trends in the project. These avenues collectively provide a thorough overview to tackle integrating a Keras model into a web page.
