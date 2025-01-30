---
title: "How to resolve a 'TypeError: load() missing 1 required positional argument: 'sess'' when loading a TensorFlow 2 Object Detection API SavedModel?"
date: "2025-01-30"
id: "how-to-resolve-a-typeerror-load-missing-1"
---
The `TypeError: load() missing 1 required positional argument: 'sess'` encountered when loading a TensorFlow 2 Object Detection API SavedModel stems from attempting to utilize a function designed for TensorFlow 1's session-based graph execution within a TensorFlow 2 eager execution environment.  My experience debugging this issue across numerous large-scale object detection projects highlighted the crucial distinction between these two TensorFlow paradigms.  TensorFlow 1 relied heavily on explicit session management, while TensorFlow 2 defaults to eager execution, eliminating the need for explicit session creation and management.  This incompatibility directly causes the error.

The solution involves adapting the loading procedure to align with TensorFlow 2's eager execution model.  This primarily means utilizing the appropriate `tf.saved_model.load` function and avoiding any code that assumes the existence of a TensorFlow 1 `tf.Session` object.

**1. Explanation:**

The TensorFlow Object Detection API's older codebases, particularly those predating TensorFlow 2's widespread adoption, often included `load()` methods expecting a TensorFlow 1 session as an argument. These methods were designed to load the model graph into a pre-created session, allowing for subsequent execution within that session's context.  In TensorFlow 2, the concept of a session is largely abstracted away.  Model loading and execution occur directly within the eager execution environment.  The `TypeError` arises because the `load()` function, expecting a `sess` argument (the TensorFlow 1 session), finds none provided in a TensorFlow 2 environment.

Therefore, the correction necessitates replacing the legacy `load()` methods with TensorFlow 2-compatible loading mechanisms.  This usually involves using `tf.saved_model.load` to directly load the SavedModel into memory, ready for immediate execution without the intermediary step of a session.  This approach seamlessly integrates with TensorFlow 2's eager execution, eliminating the need for explicit session management.  Furthermore, any code relying on session-specific operations, such as `sess.run()`, should be rewritten to directly execute TensorFlow operations in eager mode.

**2. Code Examples:**

**Example 1: Incorrect (TensorFlow 1 style):**

```python
import tensorflow as tf

# ... (previous code loading the model path) ...

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with tf.compat.v1.Session(graph=detection_graph) as sess:  # This line causes the error in TF2
    # ... (rest of the code using 'sess') ...
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # ... (further tensor retrieval and processing using 'sess.run()') ...
```

This example demonstrates the problematic approach. The creation of a `tf.compat.v1.Session` is incompatible with TensorFlow 2's eager execution, leading to the `TypeError`.

**Example 2: Correct (TensorFlow 2 style using tf.saved_model.load):**

```python
import tensorflow as tf

# ... (previous code loading the model path) ...

detect_fn = tf.saved_model.load(PATH_TO_CKPT)

# Access model inputs and outputs directly
image_tensor = detect_fn.signatures['serving_default'].inputs[0]
detection_boxes = detect_fn.signatures['serving_default'].outputs['detection_boxes']

# Inference
detections = detect_fn.signatures['serving_default'](image_tensor=image)

# Access results
boxes = detections['detection_boxes'].numpy()
# ... (further processing of the results) ...
```

This corrected example leverages `tf.saved_model.load` to load the SavedModel.  The model's inputs and outputs are accessed directly through the `signatures` attribute, and inference is performed directly without any `sess.run()` calls.  Note that the 'serving_default' signature is commonly used, but other signatures might be present in your SavedModel.


**Example 3: Correct (TensorFlow 2 style handling potential signature issues):**

```python
import tensorflow as tf

# ... (previous code loading the model path) ...

try:
    detect_fn = tf.saved_model.load(PATH_TO_CKPT)
    # ... (Proceed with Example 2's inference code) ...
except ValueError as e:
    if "No signature found with name" in str(e):
        print("Error: The SavedModel does not contain the expected signature 'serving_default'.")
        print("Please check your model export process and ensure it includes a 'serving_default' signature.")
        # ... (Handle the error appropriately, e.g., fallback mechanism or exit) ...
    else:
        raise  # Re-raise the exception if it's not related to the signature.
```

This example adds robust error handling, specifically addressing potential issues if the SavedModel doesn't contain the expected 'serving_default' signature.  In my experience, inconsistent model export processes sometimes resulted in missing or incorrectly named signatures. This code addresses that common pitfall.


**3. Resource Recommendations:**

The official TensorFlow documentation on SavedModel, the TensorFlow Object Detection API's documentation, and a good introductory textbook on TensorFlow 2 are invaluable resources.  Specifically, focusing on the sections regarding model loading and eager execution will clarify the concepts discussed here.  Additionally, exploring tutorials and examples that demonstrate loading and using pre-trained object detection models within TensorFlow 2 will provide practical guidance.  Reviewing the TensorFlow API documentation for `tf.saved_model.load` is also crucial.  Finally, understanding the differences between TensorFlow 1's graph execution and TensorFlow 2's eager execution is fundamental to resolving this type of error.
