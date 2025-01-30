---
title: "How can I upload local files for inference using SageMaker deployments?"
date: "2025-01-30"
id: "how-can-i-upload-local-files-for-inference"
---
The challenge of seamlessly integrating local file uploads for inference within a SageMaker deployment stems from the inherent separation between a client’s local environment and the cloud-based infrastructure where SageMaker models reside. Direct file system access from the deployed endpoint is intentionally restricted for security and scalability reasons. Therefore, a mechanism to transfer local files to a location accessible by the inference code is necessary. This typically involves pre-processing the files on the client side and then transmitting them, often as encoded data, to the SageMaker endpoint.

My experiences while building a custom image processing application for a medical diagnostics company highlight this process. We needed to allow clinicians to upload local medical images directly from their workstations to a model hosted on SageMaker. Instead of attempting to access local file paths from the endpoint, we opted for a pipeline involving client-side encoding, server-side decoding, and finally inference.

The core principle relies on transforming local file data into a suitable format for transmission, usually base64 encoding, and then reversing the process once the data reaches the inference container. This approach minimizes dependencies on specific file storage systems or shared network drives, promoting a more robust and scalable solution. SageMaker endpoints typically expect request bodies in JSON format, making it convenient to embed encoded file data within a structured request.

To illustrate, consider a simplified scenario involving image files. Here are three code examples outlining the client and inference endpoint code:

**Code Example 1: Client-Side File Upload and Encoding (Python)**

```python
import base64
import json
import requests

def upload_for_inference(file_path, endpoint_url):
  """Encodes a local file and sends it to a SageMaker endpoint."""
  try:
    with open(file_path, "rb") as f:
      file_bytes = f.read()
      encoded_file = base64.b64encode(file_bytes).decode("utf-8") #encode bytes to base64 string
      payload = {"image": encoded_file}
      headers = {'Content-type': 'application/json'}
      response = requests.post(endpoint_url, data=json.dumps(payload), headers=headers)

    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    return response.json()
  except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    return None
  except requests.exceptions.RequestException as e:
     print(f"Error during request: {e}")
     return None

if __name__ == '__main__':
    local_image_path = "local_image.png"
    sagemaker_endpoint_url = "https://your-sagemaker-endpoint.amazonaws.com/invocations"
    inference_result = upload_for_inference(local_image_path, sagemaker_endpoint_url)

    if inference_result:
      print("Inference result:", inference_result)
```

This client-side Python script performs several crucial steps. The function `upload_for_inference` takes the local file path and the SageMaker endpoint URL as input. It reads the file in binary format, encodes the bytes using base64, and structures it into a JSON payload, where the "image" key holds the encoded string. The script then uses the `requests` library to send the payload to the specified endpoint using an HTTP POST request. Crucially, the `response.raise_for_status()` line is included to check for errors in the HTTP response before processing, indicating a communication issue. Error handling is also incorporated to handle both file not found and connection errors. I prefer this method for error handling because it provides more clarity when debugging client-side transmission issues.

**Code Example 2: Inference Endpoint Implementation (Python using Flask)**

```python
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image #example image handling
import numpy as np

app = Flask(__name__)

@app.route('/invocations', methods=['POST'])
def invocations():
  """Handles the request, decodes the image and performs inference."""
  try:
    data = request.get_json()
    encoded_image = data['image']
    image_bytes = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_bytes))

    # Placeholder for your model inference code
    # Here we just return the size for demonstration
    image_array = np.array(image) # Convert image to numpy array
    height, width = image_array.shape[:2]
    inference_output = {"image_height": height, "image_width": width }


    return jsonify(inference_output)
  except (KeyError, base64.binascii.Error) as e:
    return jsonify({"error": f"Decoding error: {e}"}), 400
  except Exception as e:
    return jsonify({"error": f"An error occurred during inference: {e}"}), 500


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
```

This code example outlines the endpoint component using Flask, a lightweight Python web framework. The `/invocations` route handles POST requests. It decodes the base64 encoded image data received within the request body. The `base64.b64decode` function reverses the encoding process, converting the string back into bytes. This byte data is then used to construct a PIL `Image` object using `io.BytesIO`, which enables loading the byte stream as an in-memory file. Subsequently, the code would typically call a model to perform inference, which is represented here by returning the shape of the input image as a placeholder. Error handling includes both specific exceptions related to base64 decoding and general exceptions which is a good practice for any endpoint receiving external data. I included the KeyError in the exception handling because I've found that missing keys in the expected JSON payload are the cause of most endpoint errors when doing rapid prototyping. Finally, the function returns the inference result as a JSON object.

**Code Example 3: Using Multipart Form Data (Alternative Client)**

```python
import requests

def upload_for_inference_multipart(file_path, endpoint_url):
    """Uploads a file using multipart form data."""
    try:
        with open(file_path, "rb") as file:
            files = {'image': file}
            response = requests.post(endpoint_url, files=files)
        response.raise_for_status()
        return response.json()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

if __name__ == '__main__':
    local_image_path = "local_image.png"
    sagemaker_endpoint_url = "https://your-sagemaker-endpoint.amazonaws.com/invocations"
    inference_result = upload_for_inference_multipart(local_image_path, sagemaker_endpoint_url)

    if inference_result:
      print("Inference result:", inference_result)
```

This example presents an alternative approach using `requests` to send file data via multipart form data instead of a JSON payload. The file is opened in binary mode and directly sent as part of the `files` parameter in the POST request. The endpoint implementation would require adaptation to handle multipart data which is outside of the scope of this example, but the core concept remains the same which is that local files need to be transferred through a web request to the inference endpoint. The response checking remains the same. Choosing between JSON and multipart depends on specific needs; large files are sometimes better handled through multipart, while JSON offers greater structural flexibility for additional metadata. My preference is JSON for most machine learning tasks because it lends itself better to embedding non-file data or complex pre-processing parameters.

When implementing such systems, some specific resources have been invaluable. I would recommend reviewing documentation on AWS SageMaker endpoint configurations, the Flask web framework for endpoint development, and Python’s built-in `base64` and `requests` libraries for data handling and networking. Further exploration of image handling libraries like PIL and data serialization formats such as JSON are beneficial. Comprehensive error handling is critical, as network communication, encoding issues, and model failures are common in production environments. Understanding these components will significantly enhance one's ability to create a robust and functional SageMaker application using local files as inputs.
