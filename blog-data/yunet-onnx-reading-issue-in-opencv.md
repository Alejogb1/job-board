---
title: "yunet onnx reading issue in opencv?"
date: "2024-12-13"
id: "yunet-onnx-reading-issue-in-opencv"
---

Okay so you're having a problem with YUNet ONNX models and OpenCV right? Seen this rodeo before. I've wrestled with similar situations more times than I'd like to admit and it's always something slightly obscure. Let's unpack this and see what's going on.

First off YUNet is a face detection model right? Based on my past experiences it’s often implemented using ONNX for portability and OpenCV is usually the go to for any kind of computer vision tasks so that makes perfect sense that you’re trying to put these two things together. I'm gonna assume you've already downloaded a proper ONNX model and have OpenCV installed correctly. If not thats step one obviously go download the model if you have not and `pip install opencv-python` or whatever your package manager uses.

My past issues generally fall into a few categories. Usually it's either a problem with model compatibility the way OpenCV is loading the model or a mismatch between input data types or shapes. You know the usual suspect list.

Let’s talk about what happens under the hood when loading an ONNX model into OpenCV. OpenCV uses `cv2dnn.readNetFromONNX` which is essentially a wrapper around an ONNX runtime session. If you’re getting errors here it could be because your ONNX model uses operations not supported by OpenCV's build or it's a model version issue. I've had situations where a model exported using a more recent version of the ONNX specification simply refused to load in OpenCV due to the runtime not understanding what it was. It happened to me a few times when my model exported from pytorch used some op not implemented in OpenCV. So let's verify if that's your issue.

```python
import cv2
import numpy as np

try:
    net = cv2.dnn.readNetFromONNX("your_yunet_model.onnx")
    print("Model loaded successfully")

    #check if layers are available for inference
    layers_names = net.getLayerNames()
    print("Layers available for inference:")
    for name in layers_names:
        print(name)

except Exception as e:
    print(f"Error loading model: {e}")

```

Run this first. If you get an error during the `readNetFromONNX` call it's likely the model itself. Check the error message carefully. If the error mentions something about an unsupported operation it means that the issue is OpenCV not knowing how to deal with an operation inside the ONNX. If it just gives generic errors check the model version of the onnx with tools like `netron` a simple visualizer that allows you to inspect your ONNX and the different versions supported.

So what's the fix? Well first off if you're using a pre-trained model make sure its from a reliable source and specifically states compatibility with OpenCV. Check the model's documentation or any release notes. Sometimes models are intended for specific runtimes like the ONNX runtime itself and are not directly compatible with OpenCV. One way to tackle this is to try exporting the model using a lower ONNX version. If you generated the model yourself and it's not just one downloaded from somewhere make sure that you export it to a lower version like 11 or 13. Sometimes newer ONNX operators cause these type of headaches. It doesn't always work but its something you can try if you generated the model yourself.

Now if the model loads fine the next thing to check is input preprocessing. YUNet models like most face detection models expect a specific input format. Usually it's an image normalized to a particular range maybe with mean subtraction and scaled. When I had issues I often missed a normalization step.

```python
import cv2
import numpy as np

# Assuming 'frame' is your image read from somewhere like webcam or a picture
def preprocess_image(frame):
    resized_image = cv2.resize(frame,(320,320)) #or whatever size the model requires
    blob = cv2.dnn.blobFromImage(resized_image,1/255,(320,320),(0,0,0),swapRB=True,crop=False)
    return blob

# Load model (assuming it's loaded already)
net = cv2.dnn.readNetFromONNX("your_yunet_model.onnx")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cannot open video feed")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        break
    # Preprocess image
    blob_image = preprocess_image(frame)
    # Set input to the model
    net.setInput(blob_image)
    # Inference
    detections = net.forward()
    # Do something with detections

    cv2.imshow("Face Detection",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Check the shape of the blob from `cv2.dnn.blobFromImage`. You see that we are setting the scale to `1/255` to normalize the image to the `[0,1]` range which most models require. And also note how we are swapping red and blue channels as that is also a common requirement. Also check the input size and if it matches the model requirements you can find this information in the model’s documentation.

Another really sneaky issue I encountered was data type mismatch. OpenCV's `cv2.dnn.blobFromImage` by default generates a 4D blob with data type float32. And in my case some custom trained models had some different requirements like float 16. Now for you this is unlikely but lets leave it here just in case.

```python
import cv2
import numpy as np
# Load model (assuming it's loaded already)
net = cv2.dnn.readNetFromONNX("your_yunet_model.onnx")

# Assuming 'frame' is your image read from somewhere like webcam or a picture
def preprocess_image(frame, target_type=np.float32):
    resized_image = cv2.resize(frame,(320,320)) #or whatever size the model requires
    blob = cv2.dnn.blobFromImage(resized_image,1/255,(320,320),(0,0,0),swapRB=True,crop=False)
    if blob.dtype != target_type:
      blob=blob.astype(target_type)
    return blob

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cannot open video feed")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        break
    # Preprocess image
    blob_image = preprocess_image(frame,target_type=np.float32)
    # Set input to the model
    net.setInput(blob_image)
    # Inference
    detections = net.forward()
    # Do something with detections

    cv2.imshow("Face Detection",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```
In this last example I've added an optional parameter to `preprocess_image` to control the data type of the blob just in case you need to change it. It will most likely not be the issue in your case but it could be.

My advice is simple: check your model versions verify your OpenCV build if necessary and carefully inspect the input data requirements. Reading the documentation of OpenCV's dnn module and the documentation of the model and it’s requirements is key for issues like these. I would also recommend reading the original research paper for YUNet they are usually a good resource as well. This is the paper "YOLOFace: A Real-Time Face Detector" which is on arXiv. Check that you are not using some unofficial implementation.

Oh also one time I spend like three days debugging something just to find out I had my model and the config file in different folders and that I was using a completely unrelated model. That was my dumbest day programming. Let’s not let that happen again right?

So there you have it my experience with these kind of issues and the most common fixes i've had to use. Debugging computer vision stuff can be like trying to find a black cat in a dark room. You usually have to poke around a little until you find the right spot. If you still have the issue provide me with any error messages and the code you are using in more detail and the model you are trying to use. Let me know. I'll do my best to help.
