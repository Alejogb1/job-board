---
title: "How can TensorFlow models be exported to ONNX with explicit variable names?"
date: "2025-01-30"
id: "how-can-tensorflow-models-be-exported-to-onnx"
---
Successfully exporting TensorFlow models to ONNX while retaining user-defined variable names requires careful management of the TensorFlow graph structure and a clear understanding of the ONNX conversion process. The default conversion often yields generated, numerical names for model inputs, outputs, and intermediary tensors, obscuring debugging and interoperability. Through several projects involving cross-platform deployment of machine learning models, I've developed an approach that maintains explicit variable names during export. This involves utilizing the TensorFlow signature mechanism and specifically naming nodes within your TensorFlow graph construction.

The core issue stems from how TensorFlow handles graph representation and how this representation is translated into the ONNX format. TensorFlow fundamentally relies on operation names and tensor names within its computational graph. The default export process, when employing `tf2onnx.convert.from_tensorflow` from the `tf2onnx` library, maps these TensorFlow graph elements to ONNX nodes. When explicit names are not specified during TensorFlow model construction, the resulting ONNX graph carries automatically generated labels based on the order and structure of the operations rather than user-defined names. To overcome this, one must explicitly assign meaningful names to inputs, outputs, and critical intermediary tensors within the TensorFlow model, and leverage TensorFlow's SavedModel signature mechanism.

The SavedModel format preserves more than just the model weights; it encapsulates the entire computation graph, including named input and output signatures. This ensures that when you load the saved model, its inputs and outputs are associated with the predefined names. The `tf2onnx` conversion process can subsequently derive user-defined names from these signatures, resulting in a more interpretable ONNX model.

Let's illustrate this with a series of code examples. First, consider a scenario where we create a simple TensorFlow model without specific naming conventions:

```python
import tensorflow as tf
import tf2onnx

# Model without explicit naming
def build_unnamed_model():
    inputs = tf.keras.Input(shape=(10,), dtype=tf.float32)
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model_unnamed = build_unnamed_model()
tf.saved_model.save(model_unnamed, "unnamed_model")

# Convert to ONNX (will have generated names)
spec = (tf.TensorSpec((1, 10), tf.float32, name="input_tensor"),)
onnx_model, _ = tf2onnx.convert.from_saved_model("unnamed_model", input_signature=spec, output_path="unnamed_model.onnx")
```

In this first example, the input tensor is named via the signature, but the subsequent layers will have automatically generated names within the ONNX model. Inspecting this ONNX file (using tools mentioned later) would reveal names such as `dense/BiasAdd`, `dense_1/BiasAdd` for node operations instead of potentially clearer names. This is undesirable if you need to understand the flow of data and interact programmatically with specific intermediate layers of the ONNX graph.

Now, compare this with a model construction method that assigns names to key tensors:

```python
import tensorflow as tf
import tf2onnx

# Model with explicit naming using signatures
def build_named_model():
    inputs = tf.keras.Input(shape=(10,), dtype=tf.float32, name="input_feature")
    x = tf.keras.layers.Dense(32, activation='relu', name='hidden_layer')(inputs)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output_score')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Define Signature for saving
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 10), dtype=tf.float32, name='input_feature')])
    def serving_fn(input_feature):
         return model(input_feature)
    
    model.serving_fn = serving_fn

    return model


model_named = build_named_model()
tf.saved_model.save(model_named, "named_model", signatures={"serving_default": model_named.serving_fn})

# Convert to ONNX (will preserve specified names)
spec_named = (tf.TensorSpec((1, 10), tf.float32, name="input_feature"),)
onnx_model_named, _ = tf2onnx.convert.from_saved_model("named_model", input_signature=spec_named, output_path="named_model.onnx")
```

Here, the input layer is specifically named "input_feature," the first dense layer is named `hidden_layer`, and the output is named `output_score`. Critically, we've also defined a TensorFlow function called `serving_fn` that explicitly takes `input_feature` as its argument and defined a serving signature when saving the model.  This signature carries the necessary name information. Upon conversion to ONNX, the ONNX graph would now reflect these names, making the model more interpretable.

Finally, a more complex scenario involves a model with multiple inputs and outputs:

```python
import tensorflow as tf
import tf2onnx

# Model with multiple inputs and outputs with explicit naming
def build_multi_io_model():
    input1 = tf.keras.Input(shape=(5,), dtype=tf.float32, name="input_feature_1")
    input2 = tf.keras.Input(shape=(7,), dtype=tf.float32, name="input_feature_2")

    concat = tf.keras.layers.concatenate([input1, input2])
    x = tf.keras.layers.Dense(64, activation='relu', name="combined_layer")(concat)
    output1 = tf.keras.layers.Dense(3, activation='softmax', name='output_class_probs')(x)
    output2 = tf.keras.layers.Dense(1, activation='linear', name='output_regression_value')(x)

    model = tf.keras.Model(inputs=[input1, input2], outputs=[output1, output2])

    # Define signature for saving with named inputs
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32, name="input_feature_1"),
        tf.TensorSpec(shape=(None, 7), dtype=tf.float32, name="input_feature_2")
    ])
    def serving_fn(input_feature_1, input_feature_2):
        return model([input_feature_1, input_feature_2])
        
    model.serving_fn = serving_fn
    return model


model_multi_io = build_multi_io_model()
tf.saved_model.save(model_multi_io, "multi_io_model", signatures={"serving_default": model_multi_io.serving_fn})


# Convert to ONNX (will preserve specified names)
spec_multi = (tf.TensorSpec((1, 5), tf.float32, name="input_feature_1"), tf.TensorSpec((1, 7), tf.float32, name="input_feature_2"))

onnx_model_multi, _ = tf2onnx.convert.from_saved_model("multi_io_model", input_signature=spec_multi, output_path="multi_io_model.onnx")
```

In this final example, we demonstrate the process with multiple inputs and outputs. Notice that again, a corresponding signature is defined before saving the model. The `input_signature` passed to the converter corresponds to the signature of the exported `SavedModel`.  This makes the corresponding ONNX graph much easier to programmatically manipulate.

In practice, the naming process should be systematically applied to every relevant layer in your model, particularly the inputs and outputs of each module. Additionally, while `tf2onnx` attempts to infer names based on the TensorFlow graph and signatures, explicit names always provide more reliable control over the final ONNX graph.

For inspecting the ONNX files after exporting, I'd highly recommend using tools such as Netron. Netron can visually render the structure of an ONNX graph and expose details, allowing you to confirm the names have been correctly carried over. Furthermore, ONNX Runtime provides APIs for programmatically interacting with ONNX graphs that will benefit substantially from consistent naming conventions.  Documentation for these tools can be readily found through their respective project webpages. The `onnx` python package provides capabilities to inspect and modify onnx graph structure, also detailed in its readily available documentation. These resources and related tutorials should solidify one's ability to consistently export named TensorFlow models to ONNX. Through careful naming of TensorFlow models and explicit signature definition, a developer gains far more control over the final ONNX graph representation, making debugging, deployment, and cross-platform interoperability easier to achieve.
