---
title: "How can ZED stereo cameras be used for object detection with TensorFlow?"
date: "2025-01-30"
id: "how-can-zed-stereo-cameras-be-used-for"
---
Stereo vision, specifically with the ZED camera, offers inherent advantages for object detection over monocular approaches due to its ability to directly provide depth information. This depth data, when integrated with the ZED's color imagery, significantly enhances the accuracy and robustness of object detection models, especially in situations with varying lighting or partial occlusions, which I've consistently encountered in my work with robotics.

The process of leveraging a ZED camera for object detection using TensorFlow typically involves several distinct steps. First, the raw image data and corresponding depth maps captured by the ZED are accessed using the ZED SDK, which is a library provided by Stereolabs. These data streams then need preprocessing to prepare them for input into a TensorFlow model. This often means aligning and normalizing the color images and depth maps, a step crucial for maintaining spatial consistency. After preprocessing, the data is fed to a pre-trained object detection model, or one that I have trained myself. The model's output, bounding boxes and class labels, can then be refined using the depth information to add a spatial dimension or filter detections based on a distance threshold. I have found this last step particularly useful to remove background noise.

Let's delve into the practical implementation. I will focus on three primary stages, starting with data acquisition and preprocessing, then transitioning to model inference and ending with the integration of depth. Each step is crucial for achieving good results.

First, accessing the ZED camera and retrieving synchronized color and depth images requires interacting with the ZED SDK's API. In Python, this typically looks like this:

```python
import pyzed.sl as sl
import numpy as np
import cv2

def get_zed_data():
    # Create a ZED camera object
    zed = sl.Camera()

    # Define camera configuration
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error opening camera, exit program.")
        exit()

    runtime_parameters = sl.RuntimeParameters()

    # Prepare sl.Mat for images
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT) # Retrieve left color image
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH) # Retrieve depth map
            
            # Convert to numpy array
            image_np = image_zed.get_data()
            depth_np = depth_zed.get_data()
            
            # Process data further here (e.g., pass to model inference function)
            process_data(image_np, depth_np)

            if cv2.waitKey(1) == ord('q'):
                break
            
    zed.close()

def process_data(image, depth):
     #Placeholder for processing, this might include
     #resizing, normalization, and other preprocess steps.
     cv2.imshow("Left Camera", image)
     cv2.imshow("Depth Map", depth/1000.0) #depth values are in mm
     pass
     

if __name__ == '__main__':
    get_zed_data()
```

In this code snippet, I've outlined the basic ZED initialization and the retrieval of synchronized color and depth frames. The ZED SDK provides the functionality to obtain these two streams concurrently, and after converting them to NumPy arrays, I display them for verification. Note that the depth values from the ZED are in millimeters; therefore a division by 1000 was necessary to visualize it as a depth map. The `process_data` function represents the next step in the pipeline.

Following data acquisition, preprocessing for the TensorFlow model is required. This often entails resizing images to match the input dimensions of the model and normalizing pixel values to a specific range. A vital addition here for stereo data is the alignment of the color and depth images. I typically do this via the intrinsic and extrinsic parameters of the camera as provided by the ZED SDK, though in many cases the ZED provides an aligned depth image automatically.

```python
import tensorflow as tf
import cv2
import numpy as np

def preprocess_data(image_np, depth_np, target_size=(300, 300)): #example size
    # Resize images to target size
    image_resized = cv2.resize(image_np, target_size)
    depth_resized = cv2.resize(depth_np, target_size, interpolation=cv2.INTER_NEAREST) # Nearest neighbor for depth
    
    # Normalize pixel values for the image
    image_normalized = (image_resized/255.0)
     
    #Depth normalizaton might be required, but I'll do it for now just in case it is too noisy
    depth_normalized = depth_resized / np.max(depth_resized) if np.max(depth_resized) > 0 else depth_resized #Avoid NaN

    # Expand dimensions to have a batch dimension 
    image_input = np.expand_dims(image_normalized, axis=0).astype(np.float32)
    depth_input = np.expand_dims(depth_normalized, axis=0).astype(np.float32)

    return image_input, depth_input

if __name__ == '__main__':
    # Dummy data for testing
    image_dummy = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    depth_dummy = np.random.randint(0, 4000, size=(720, 1280), dtype=np.uint16)

    image_processed, depth_processed = preprocess_data(image_dummy, depth_dummy)

    print("Image Input shape:", image_processed.shape)
    print("Depth Input shape:", depth_processed.shape)
```
Here, I've demonstrated image resizing, normalization, and the expansion of the array dimensions to accommodate a batch size of one, which is a requirement for most TensorFlow models. I've also utilized nearest neighbor interpolation when resizing the depth map to preserve the values, as opposed to interpolating them. The depth map normalization here is done assuming a global scaling, but the optimal method might vary depending on the specific model and the range of depth it expects. Finally, for testing purposes I've added a dummy input and print statements, showing the tensor dimensions after processing.

Finally, after the images have been preprocessed, they can be fed into a TensorFlow model. This step involves loading a model, making inferences, and post-processing the model's output.  I'll focus here on a model that takes in images and depth maps individually, then combines them at a later step.

```python
import tensorflow as tf
import numpy as np

def load_model(model_path):
    # Example loading of an existing model, adjust this for your model architecture
    model = tf.saved_model.load(model_path) #using a saved model
    return model

def perform_detection(model, image_input, depth_input, depth_threshold=2000): #2 meters
    # Run the model on input data, adjust function call per your model
     #Example of feeding image and depth in as inputs separately.
    outputs = model(image_input, depth_input)  # Assumes a function style model
    # The output will vary based on the detection model you use.
    # For a common object detection model output would typically include bounding boxes and class labels.
    # Assumes it looks like [[x1,y1,x2,y2,class_id, confidence] ... ]

    # Process the detection outputs to get detections
    bounding_boxes = outputs['detection_boxes'][0].numpy()
    classes = outputs['detection_classes'][0].numpy().astype(np.int32)
    scores = outputs['detection_scores'][0].numpy()


    filtered_detections = []

    for i, score in enumerate(scores):
        if score > 0.5: # minimum confidence threshold
           x1, y1, x2, y2 = bounding_boxes[i] 
           depth_value = depth_input[0,int((y1+y2)/2),int((x1+x2)/2)] # average depth of the bounding box
           if depth_value > depth_threshold:
               continue # skip object based on distance

           filtered_detections.append([x1,y1,x2,y2,classes[i],score, depth_value])
    return filtered_detections

if __name__ == '__main__':
    # Dummy preprocessed data for testing
    dummy_image_input = np.random.rand(1,300,300,3).astype(np.float32)
    dummy_depth_input = np.random.rand(1,300,300).astype(np.float32)
    
    # Path to your SavedModel
    model_path = 'path/to/saved_model'
    model = load_model(model_path)
    
    detections = perform_detection(model, dummy_image_input, dummy_depth_input)
    for detection in detections:
        x1,y1,x2,y2,class_id,confidence, depth = detection
        print(f"Detection at ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f}), Class: {class_id}, Confidence: {confidence:.2f}, Depth:{depth:.2f}")
```
In this example, the `load_model` function loads a pre-trained or a trained object detection model from disk using TensorFlow’s `saved_model` format. The `perform_detection` function takes preprocessed image and depth data and performs inference. It then processes the model output to extract bounding boxes, class labels, and confidence scores. Finally, I have added a simple filtering mechanism based on depth, discarding detections that are outside a defined threshold. The final part is an example of how to print out the detections to the console. Again, this is an example and the exact processing will depend heavily on the specific model.

To further enhance the accuracy and performance of this process, I recommend consulting resources detailing object detection using TensorFlow, specifically focusing on utilizing depth information. Resources like “TensorFlow Object Detection API” tutorials can provide in-depth information on training and using models. Additionally, research articles focusing on stereo vision and depth-aware object detection algorithms will improve understanding of the underlying principles and practical implementation techniques. Finally, Stereolabs provides extensive documentation on their ZED SDK. These are the three resources that I have found most beneficial over time and that have been proven to provide valuable insights.

In conclusion, object detection using a ZED camera and TensorFlow requires careful consideration of data acquisition, preprocessing, and model inference. Utilizing the ZED camera's stereo capabilities to provide depth information offers substantial performance improvements over monocular approaches. My own experience suggests that iterative testing and refinement of each stage is crucial for achieving the desired accuracy and robustness of the object detection system.
