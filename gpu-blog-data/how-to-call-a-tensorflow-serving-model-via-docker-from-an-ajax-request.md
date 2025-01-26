---
title: "How to call a TensorFlow Serving model via Docker from an AJAX request?"
date: "2025-01-26"
id: "how-to-call-a-tensorflow-serving-model-via-docker-from-an-ajax-request"
---

The crux of deploying TensorFlow models for web applications often involves bridging the gap between the client-side's asynchronous requests and the server-side's machine learning inference. This frequently necessitates communicating with a model served through Docker, and achieving this via AJAX requests requires careful attention to data formatting, API endpoints, and network configurations.

From my experience building multiple production-ready systems, I've found the most reliable method involves leveraging TensorFlow Serving's REST API, which neatly circumvents many of the complexities that arise with other communication protocols. The typical workflow comprises three main stages: preprocessing data on the client, sending a JSON payload to the server, and then post-processing the returned inference result. Let's break down each of these aspects along with illustrating some typical implementation patterns using both Javascript and Python.

Firstly, data preprocessing occurs client-side, ensuring data sent to the model aligns with its expected input format. This involves converting user input, whether text, images, or numerical values, into the appropriate JSON structure. It's absolutely critical the input shape matches the model definition. For example, a model expecting a batch of 128x128 RGB images will require the client to resize, format, and potentially normalize the user-provided image before transmission. The JSON structure needs to correspond to TensorFlow Serving's expected input structure. This is generally a dictionary with a key "inputs" and an associated list (for batch inputs) of numerical values or base64-encoded images.

Secondly, the client-side AJAX call must use the correct URL and HTTP method, specifically, a POST request to the specified endpoint (generally "/v1/models/<model_name>/versions/<version_number>:predict"). The content type should be set to "application/json," and the JSON encoded data, created in step one, is included in the request body. Correctly configuring CORS on the server is vital to avoid cross-origin request issues that can impede successful interaction. A properly configured response from the server will be a 200 status code with a JSON body containing the model's predictions.

Lastly, once the client receives the response, the JSON prediction data needs to be post-processed to be comprehensible to the user. This might involve converting predicted probability vectors to labels for a classifier, mapping bounding box coordinates in an object detector, or rendering generated text.

Let's look at the required code implementations:

**Example 1: Client-side (Javascript) with Text Input**

This snippet demonstrates how to make a prediction request using Javascript with a text input.

```javascript
function predictText(inputText) {
  const modelName = "my_text_model";
  const modelVersion = "1";
  const url = `/v1/models/${modelName}/versions/${modelVersion}:predict`;

  const payload = {
    inputs: [[inputText]],  // Assuming the model expects a batch of single string
  };

  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
  .then((response) => {
      if(!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("Model Prediction:", data.outputs);
      // Process predictions here
    })
    .catch((error) => {
      console.error("Error during prediction:", error);
    });
}

// Example usage
const textInput = "This is a sample text input.";
predictText(textInput);

```

In this first example, note the explicit specification of the model name and version within the URL. The payload object has to conform to TensorFlow Serving's input format, as a nested list for a batch of a single text input. Error handling is also implemented, along with decoding the JSON response before further processing the model's output.

**Example 2: Client-side (Javascript) with Image Input**

Here, an image is processed and sent to the model in base64 encoding.

```javascript
async function predictImage(imageElement) {
  const modelName = "my_image_model";
  const modelVersion = "1";
    const url = `/v1/models/${modelName}/versions/${modelVersion}:predict`;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  const targetSize = 128 // Model input size.
    canvas.width = targetSize;
    canvas.height = targetSize;

  ctx.drawImage(imageElement, 0, 0, targetSize, targetSize)
  const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];

    const payload = {
    inputs: [[base64Image]] // Model expects a single image, formatted as base64
    };


  try {
    const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if(!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
       console.log("Model Prediction:", data.outputs);
      // Process prediction data
  } catch (error) {
    console.error("Error during image prediction:", error);
  }
}

// Example usage:
const imgElement = document.getElementById('my_image');
imgElement.onload = () => predictImage(imgElement);
```

This example first uses the HTML5 canvas to resize the input image to the size expected by the model. The base64 encoded image data is then included in the request payload. Error handling and asynchronous request processing are implemented via `async/await`. It is important to note that base64 encoding will lead to significant size increases in the JSON payload and larger bandwidth consumption.

**Example 3: Server-Side (Python) for Simple Testing**

This Python snippet serves as a simple test to verify if the server is properly configured:

```python
import requests
import json

def test_prediction():
    url = "http://localhost:8501/v1/models/my_test_model/versions/1:predict" # Adjust the URL if needed.
    headers = {'Content-type': 'application/json'}
    payload = {
        "inputs": [[1.0, 2.0, 3.0, 4.0]]  # Example numerical input
        }
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        print("Prediction successful")
        print("Response:", response.json())
    else:
        print(f"Error: Status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_prediction()
```

This Python script utilizes the `requests` library to send a POST request directly to the TensorFlow Serving endpoint. It can be used to directly test if the TensorFlow Serving instance is functioning and accepting requests. The data structure being sent (in json) mirrors what the javascript client sends in Example 1.

Implementing this type of setup requires meticulous planning and coordination between client and server-side logic. Furthermore, one needs to be aware of performance implications in terms of request size, and batching of predictions. Batching helps increase the throughput of the model since Tensorflow is highly optimized for tensor calculations.

For further study, I'd recommend consulting resources covering TensorFlow Serving's REST API documentation, particularly focusing on the JSON input format for different model types (e.g., classification, regression, object detection). Also, research materials on asynchronous programming in JavaScript for handling AJAX requests efficiently would prove invaluable. Furthermore, delving into Docker networking can be crucial when dealing with cross-container communications. Resources dedicated to CORS configurations are also important for correctly handling cross-origin requests in web browsers.
