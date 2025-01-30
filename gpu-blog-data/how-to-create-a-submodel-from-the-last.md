---
title: "How to create a submodel from the last layers of a Keras model?"
date: "2025-01-30"
id: "how-to-create-a-submodel-from-the-last"
---
TensorFlow's Keras API provides a straightforward mechanism for extracting a submodel from the terminal layers of an existing, trained model. This is a powerful technique for transfer learning, feature extraction, or building ensemble models, allowing reuse of learned representations without retraining the foundational layers. Over my past projects dealing with image classification and natural language processing, I've frequently used this methodology to leverage pre-trained models, and I have encountered specific approaches which are consistently effective. The critical point involves defining the input and output tensors of the submodel, effectively isolating the desired portion of the computational graph.

The process hinges on accessing the layers of the parent model through its `.layers` attribute, then identifying which layers constitute the "last" part we wish to extract. This determination depends entirely on the specific architecture and requirements. Typically, one would select layers representing high-level feature encodings, often located in the classification or regression head after the main feature extraction backbone. Once identified, we can create a new Keras `Model` object by specifying the input tensor of the parent model and the output tensor of the chosen final layer or layers.

Specifically, the process involves the following steps:

1. **Load or Create the Base Model:** This initial step might involve loading a pre-trained model or creating a model from scratch. The key point here is to have a Keras `Model` object with layers that can be accessed and manipulated.

2. **Inspect Layer Names and Structure:** It is critical to understand the architecture of the base model using `model.summary()`. This helps visually identify the correct ending layers. Pay attention to the naming convention of the layers, as this will be used to reference them.

3. **Select the Final Layer(s):** From the `summary`, choose the layer or layers that represent the terminal portion of the base model you want to re-use. Note the name and position of these layers. You may need more than one if the desired submodel requires it, such as multi-head outputs, but most of the time you will use just a single layer as the submodel end.

4. **Create Input Tensor:** The new submodel needs an input tensor from the start of the original model. This can be accessed with `base_model.input`.

5. **Create Output Tensor:** Obtain the output tensor from the final layer(s) you identified. This is typically accessed using the layer name: `base_model.get_layer(layer_name).output`. This will be used as the output of our new model.

6. **Construct the Submodel:** Finally, create the new submodel, providing both input and output tensors: `submodel = tf.keras.Model(inputs=base_model.input, outputs=final_layer_output)`. The `submodel` will now represent only the selected part of the original computational graph.

7. **Freeze Layers (Optional):** If you want to perform transfer learning or feature extraction, the layers in your new submodel may need to have their weights frozen during training with `submodel.trainable = False`. This prevents the pre-trained weights from being updated further.

Here are three code examples to illustrate:

**Example 1: Extracting the Classification Head of a CNN**

In this example, we assume we have a convolutional neural network (CNN) for image classification. We want to extract only its classification layers for a different classification task.

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model

# Create a sample CNN for demonstration
input_tensor = Input(shape=(28, 28, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu', name="intermediate_dense")(x) # Named layer
output_tensor = Dense(10, activation='softmax', name="classification_head")(x)  # Named layer
base_model = Model(inputs=input_tensor, outputs=output_tensor)

# Extract the classification head submodel
classification_head_output = base_model.get_layer("classification_head").output
classification_submodel = Model(inputs=base_model.input, outputs=classification_head_output)

# Verify submodel
classification_submodel.summary()
```

This code creates a base CNN model with an intermediate fully connected layer followed by a classification head. We then create a submodel consisting only of the classification head. Note the use of named layers to access them in our code. The output of the `classification_submodel.summary()` will display only the layers required.

**Example 2: Extracting the Embedding Layer from an NLP Model**

Here, we assume a textual model that produces an embedding vector. We wish to extract just the embedding layer.

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, LSTM, Dense
from tensorflow.keras.models import Model

# Create a sample NLP model
input_tensor = Input(shape=(100,), dtype='int32')
x = Embedding(input_dim=10000, output_dim=128, input_length=100, name="embedding_layer")(input_tensor) # Named layer
x = LSTM(64)(x)
output_tensor = Dense(1, activation='sigmoid')(x)
base_model = Model(inputs=input_tensor, outputs=output_tensor)

# Extract the embedding layer
embedding_output = base_model.get_layer("embedding_layer").output
embedding_submodel = Model(inputs=base_model.input, outputs=embedding_output)

# Verify submodel
embedding_submodel.summary()
```

In this case, we are focusing on the embedding layer. We can create a new model that accepts the original text sequences as input and outputs the embedding vector created by the `Embedding` layer. The output of `embedding_submodel.summary()` will display that it only contains the input and the embedding layer.

**Example 3: Extracting Multiple Output Layers**

This example demonstrates extracting two final dense layers, if we require that from a more complicated classification task.

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# Create a sample multi-output model
input_tensor = Input(shape=(128,))
x = Dense(64, activation='relu')(input_tensor)
output1_tensor = Dense(10, activation='softmax', name="classification_output")(x)
output2_tensor = Dense(3, activation='sigmoid', name="regression_output")(x)

base_model = Model(inputs=input_tensor, outputs=[output1_tensor, output2_tensor])

# Extract the final two output layers into a new submodel
submodel_outputs = [base_model.get_layer("classification_output").output,
                   base_model.get_layer("regression_output").output]
final_output_submodel = Model(inputs=base_model.input, outputs=submodel_outputs)

# Verify the submodel
final_output_submodel.summary()
```

Here, the `base_model` has two outputs that share an intermediate layer. The new submodel contains both output layers as part of its final output. We capture both using a list of tensors and construct the new model accordingly. Again, the submodel summary will only show layers required for the output.

For additional learning resources, I highly recommend referring to the official TensorFlow documentation on the Keras API, particularly the sections on model creation, layer access, and transfer learning. Also, there are numerous tutorials and examples on model subclassing and functional model construction in the TensorFlow tutorials. Finally, reading papers in the specific area you are working with will usually provide useful insights into architecture selection. By examining these resources and experimenting with the concepts discussed, one can develop a strong understanding of how to effectively create submodels for diverse deep learning tasks.
