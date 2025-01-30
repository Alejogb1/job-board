---
title: "How can I perform semantic segmentation on video using a TensorFlow 2 saved model?"
date: "2025-01-30"
id: "how-can-i-perform-semantic-segmentation-on-video"
---
Semantic segmentation of video, when leveraging a pre-trained TensorFlow 2 model, presents a unique challenge compared to still images due to the temporal dimension. While the core model architecture remains consistent, the data pipeline and processing must account for sequential frames. Successfully applying a segmentation model trained on static images to video requires careful consideration of frame processing, potential temporal consistency enhancements, and efficient resource management.

The primary procedure involves loading the TensorFlow 2 saved model and then iterating through the video frames, performing inference on each frame independently. This, in its most basic form, will yield a segmented output for each frame. The efficacy, however, depends on input preprocessing consistent with the training data of the pre-trained model and often benefits from post-processing techniques. Further, one can also enhance the overall output with temporal filtering techniques.

Let's assume I have experience with a semantic segmentation model trained on the Cityscapes dataset, saved as `cityscapes_model` in a TensorFlow SavedModel format. This model, in my hypothetical projects, would accept images of shape (256, 512, 3) and output a tensor of shape (256, 512, num_classes), where num_classes is the number of segmentation classes (34 for Cityscapes). My approach to video segmentation using such a model involves the following process:

1.  **Video Loading and Frame Extraction:** Utilize a library like OpenCV or moviepy to load the video file and extract frames. I've typically chosen OpenCV for its performance and native integration with NumPy arrays.
2.  **Frame Preprocessing:** Reshape, normalize, and preprocess extracted frames to match the input format expected by the saved model. This step is critical for model accuracy and avoiding erroneous results.
3.  **Model Inference:** Perform prediction on each preprocessed frame using the loaded saved model.
4.  **Output Post-processing:**  Convert the model's output logits (or probabilities) to segmentation masks. Color-coding the masks or overlaying them back onto the original frames can be part of this step.
5.  **Output Saving/Display:** Combine the processed frames to form a segmented video or display them in real-time if desired.

Here are examples demonstrating the core concepts:

**Example 1: Frame Extraction and Preprocessing**

```python
import cv2
import tensorflow as tf
import numpy as np

def preprocess_frame(frame, target_height=256, target_width=512):
    """
    Resizes and normalizes a frame for the model.
    Assumes input frame is an OpenCV NumPy array.
    """
    resized_frame = cv2.resize(frame, (target_width, target_height))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.astype(np.float32) # Explicit type

def process_video(video_path, model_path):
    """
    Loads video, preprocesses frames and returns a generator for inference.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise IOError("Cannot open video file")

    model = tf.saved_model.load(model_path)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
           break # End of video
        preprocessed_frame = preprocess_frame(frame)
        yield preprocessed_frame, frame #Yield both for later overlay
    cap.release()


# Example Usage
video_path = 'my_video.mp4'
model_path = 'cityscapes_model'
processed_frames = process_video(video_path, model_path)
# Note: 'processed_frames' now is a generator ready for the next example.
```
*Commentary:* This code defines functions for frame preprocessing and video loading. The `preprocess_frame` function resizes the frame to the expected input dimensions and normalizes pixel values to a range between 0 and 1. The `process_video` function uses OpenCV to load the video, iteratively extracts frames, preprocesses them using the previous function and yields the processed frame along with its original unproccessed counterpart for overlaying the mask on the original image later. Crucially, the generator helps in memory management for large videos. The `np.float32` cast guarantees type consistency with TensorFlow during inference. The 'yield' keyword turns the function into a generator, allowing memory efficient iteration.

**Example 2: Performing Inference and Creating Mask**

```python
def create_segmentation_mask(model, preprocessed_frame):
    """
    Performs segmentation inference on a single preprocessed frame.
    Returns the segmented mask.
    """
    input_tensor = tf.expand_dims(preprocessed_frame, axis=0)  # Add batch dim
    predictions = model(input_tensor)
    mask = tf.argmax(predictions, axis=-1)
    mask = tf.squeeze(mask, axis=0)  # Remove batch dim
    return mask.numpy().astype(np.uint8)

def overlay_mask(original_frame, segmentation_mask, color=(0, 255, 0)):
  """
  Overlays the segmentation mask (single class) on the original image
  """
  mask_colored = np.zeros_like(original_frame)
  mask_colored[segmentation_mask == 1] = color
  overlayed_frame = cv2.addWeighted(original_frame, 1, mask_colored, 0.5, 0)
  return overlayed_frame


# Example Usage (continuing from previous example)
try:
    for preprocessed_frame, frame in processed_frames:
        mask = create_segmentation_mask(model, preprocessed_frame)
        colored_frame = overlay_mask(frame,mask)

        cv2.imshow('Segmented Frame', colored_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error processing frame {e}")

cv2.destroyAllWindows()

```
*Commentary:* This section defines the `create_segmentation_mask` function, which takes the preprocessed frame, adds a batch dimension, performs model inference, extracts the predicted class indices (the segmentation mask), and removes the batch dimension from the mask, converting the prediction from logits/probabilities to the actual predicted class for each pixel. I also included a function `overlay_mask` that takes the original frame and the generated mask, creates an overlay by coloring the detected pixels, then weighting and adding it to the original frame. The overlay operation in `cv2.addWeighted` results in a blending of the original image with the mask that was generated.  The main loop iterates through the generated preprocessed frames from the previous example to perform model prediction, generate the mask and overlay and display the result until 'q' is pressed or the end of video has been reached. Exception handling is included for error detection during the display process.

**Example 3: Saving the segmented video:**

```python
# Continuing previous examples.
output_video_path = 'segmented_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

try:
    first_frame, _ = next(process_video(video_path, model_path))
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width,height)) #Assume 30FPS

    processed_frames_reinit = process_video(video_path, model_path)
    for preprocessed_frame, frame in processed_frames_reinit:
        mask = create_segmentation_mask(model, preprocessed_frame)
        colored_frame = overlay_mask(frame,mask)
        out.write(colored_frame)
except Exception as e:
    print(f"Error saving video {e}")

finally:
    if 'out' in locals():
        out.release()
```

*Commentary:* This final code snippet handles the process of saving the segmented video to a file. The first frame is used to get the height and width of the video which then are used to initialize the video writer instance. The code iterates through the same generated video frames from before and applies the previous logic to generate the masks and apply overlays. Instead of displaying the segmented frames, each processed frame is written to the output video using the `out.write()` function. Exception handling and releasing the writer are included for error prevention and clean-up. Note how the `process_video` generator must be reinitialized because the previous loop consumed it. The `locals()` check is to ensure that `out.release()` does not throw an error if the video writer was not initialized because of a prior error.

**Resource Recommendations**

For individuals seeking to deepen their understanding of this process, I suggest consulting resources focused on the following:

*   **TensorFlow Documentation:** The official TensorFlow documentation provides an excellent foundation for understanding the API, specifically relating to model saving, loading, and inference.
*   **OpenCV Documentation:** The official OpenCV documentation offers comprehensive guides on video processing, including frame extraction, image manipulation, and video writing capabilities.
*   **Computer Vision Textbooks:** General computer vision textbooks often cover topics like semantic segmentation and image processing fundamentals.
*   **Online Tutorials:** Numerous online tutorials offer practical demonstrations and code examples related to image processing, deep learning, and video analytics. While I do not provide links, I suggest using relevant keywords like 'TensorFlow 2 video segmentation,' 'OpenCV video processing,' and 'semantic segmentation basics' when searching.
*  **Academic Papers:** Research papers published in computer vision conferences (e.g., CVPR, ICCV, ECCV) present the latest advancements in the field. Focusing on papers that concern temporal processing in segmentation tasks might be beneficial if temporal consistency becomes a requirement.

This response illustrates a foundational approach to video segmentation utilizing a TensorFlow 2 SavedModel. The actual implementation will necessitate fine-tuning according to the specific model's architecture, dataset, and application requirements. Furthermore, additional research into techniques like temporal smoothing and optical flow could enhance the robustness and smoothness of results for video.
