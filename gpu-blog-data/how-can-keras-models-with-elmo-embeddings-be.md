---
title: "How can Keras models with ELMo embeddings be saved and loaded?"
date: "2025-01-30"
id: "how-can-keras-models-with-elmo-embeddings-be"
---
Pre-trained ELMo embeddings from TensorFlow Hub offer significant contextual understanding for text-based models, yet their integration with Keras, especially when considering saving and loading model architectures, introduces complexities beyond simple `model.save()` procedures. I’ve encountered these challenges firsthand in a prior sentiment classification project, where the model’s deployment pipeline demanded reliable serialization and deserialization of the entire setup, including the ELMo embedding layer.

The central issue lies in the nature of TensorFlow Hub modules. While they are TensorFlow objects, they are external to the Keras model's native structure. When Keras saves a model using `model.save()`, it serializes the layers defined within that specific model instance. A direct embedding layer using ELMo from TensorFlow Hub is not automatically included in this process because it exists as a reference to a module rather than a defined layer. This means that simply loading a saved Keras model won't instantiate the ELMo module, leading to errors during prediction or further training unless the ELMo layer is explicitly managed.

To properly save and load a Keras model with an ELMo layer, we must employ a strategy that preserves the ELMo module’s definition in conjunction with the Keras model's architecture. Two primary approaches prove effective: subclassing the Keras Model class to manage the ELMo module or saving both the Keras model's weights separately from the functional layer utilizing a module. I prefer the latter as it maintains a cleaner structure.

**Approach 1: Subclassing Keras Model**

The first approach centers on subclassing Keras' `Model` class. By defining a custom model, we can directly manage the ELMo module within the model's build and call functions. This allows for explicit inclusion of the module when the model's architecture is built, ensuring it is always associated with the Keras structure.

```python
import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras.layers import Input, Lambda, Dense

class ELMoModel(keras.Model):
    def __init__(self, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.elmo = hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=trainable)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        elmo_output = self.elmo(tf.squeeze(tf.cast(inputs, tf.string), axis=1), as_dict=True)['elmo']
        output = self.dense(elmo_output)
        return output


#Define model and inputs
input_text = Input(shape=(1,), dtype=tf.string)
model = ELMoModel()
output = model(input_text)

#Instantiate and Compile Model
model = keras.Model(inputs=input_text,outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Example Training
data = tf.constant([["This is a test."],["Another sentence."]])
labels = tf.constant([[0],[1]], dtype = tf.float32)
model.fit(data,labels,epochs=1)

#Saving
model.save("elmo_model_subclassed.h5")

#Loading
loaded_model = keras.models.load_model("elmo_model_subclassed.h5",
    custom_objects={'ELMoModel': ELMoModel})

#Test
test_input = tf.constant([["This model works"]])
print(loaded_model.predict(test_input))
```

In this code, I create a `ELMoModel` class inheriting from Keras `Model`. The constructor initializes the ELMo layer as a TensorFlow Hub KerasLayer and also a downstream layer. In the `call` function, I feed the input data through the ELMo layer, process the output, and then pass it through a downstream layer. This method encapsulates ELMo within the custom model, enabling direct model saving and loading. I found this approach to be very direct, and it works by saving the entire class in the h5 format.

**Approach 2: Saving Weights and Functional Layer**

Alternatively, one can save the weights of the Keras model separately from the functional layer utilizing a tf.hub module. This allows for more flexible model construction and saves the keras weights into the standard h5 format and eliminates the need to subclass the base Keras Model class.

```python
import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras.layers import Input, Lambda, Dense


def elmo_embedding(x):
  elmo_model = hub.KerasLayer("https://tfhub.dev/google/elmo/3",
                        trainable=False)
  embeddings = elmo_model(tf.squeeze(tf.cast(x, tf.string), axis=1), as_dict=True)['elmo']
  return embeddings


#Define model and inputs
input_text = Input(shape=(1,), dtype=tf.string)
elmo_layer = Lambda(elmo_embedding)(input_text)
output = Dense(1, activation='sigmoid')(elmo_layer)

#Instantiate and Compile Model
model = keras.Model(inputs=input_text,outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Example Training
data = tf.constant([["This is a test."],["Another sentence."]])
labels = tf.constant([[0],[1]], dtype = tf.float32)
model.fit(data,labels,epochs=1)

#Saving weights
model.save_weights("elmo_model_weights.h5")
#Saving model
import json
model_json = model.to_json()
with open("elmo_model_architecture.json", "w") as json_file:
  json.dump(model_json, json_file)

#Loading model architecture and weights
with open("elmo_model_architecture.json", 'r') as json_file:
    loaded_model_json = json.load(json_file)
loaded_model = keras.models.model_from_json(loaded_model_json,custom_objects={"elmo_embedding":elmo_embedding})
loaded_model.load_weights("elmo_model_weights.h5")

#Test
test_input = tf.constant([["This model works"]])
print(loaded_model.predict(test_input))
```
Here, I establish a `elmo_embedding` function to encapsulate the ELMo layer and integrate it with the Keras model using a `Lambda` layer. I save the model's weights separately using `save_weights()` and then use `to_json()` to get the model architecture which can be reloaded from the saved JSON file. Upon loading, I rebuild the model architecture using `model_from_json()` and use custom objects parameter to pass the embedding function and then load the saved weights using `load_weights()`. In my experience, I've found this approach highly adaptable, particularly when the embedding layer requires further processing or needs to be modular within a larger model framework.

**Approach 3: Using `tf.saved_model` format**

Finally, I want to also mention that you can save the entire model, including the hub module, as a `tf.saved_model` format. This format is TensorFlow's recommended way to save models because it's meant to be a comprehensive way to save all parts of a model.

```python
import tensorflow as tf
import tensorflow_hub as hub
import keras
from keras.layers import Input, Lambda, Dense


def elmo_embedding(x):
  elmo_model = hub.KerasLayer("https://tfhub.dev/google/elmo/3",
                        trainable=False)
  embeddings = elmo_model(tf.squeeze(tf.cast(x, tf.string), axis=1), as_dict=True)['elmo']
  return embeddings


#Define model and inputs
input_text = Input(shape=(1,), dtype=tf.string)
elmo_layer = Lambda(elmo_embedding)(input_text)
output = Dense(1, activation='sigmoid')(elmo_layer)

#Instantiate and Compile Model
model = keras.Model(inputs=input_text,outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Example Training
data = tf.constant([["This is a test."],["Another sentence."]])
labels = tf.constant([[0],[1]], dtype = tf.float32)
model.fit(data,labels,epochs=1)

#Saving whole model with saved_model format
tf.saved_model.save(model, 'elmo_saved_model')

#Loading
loaded_model = tf.saved_model.load('elmo_saved_model')
#Test
test_input = tf.constant([["This model works"]])
loaded_model = loaded_model.signatures["serving_default"]
print(loaded_model(test_input))

```
In this last example, we simply utilize the `tf.saved_model.save` function to save our entire model object. Upon loading, the model is loaded back as an entire object, complete with all of its variables and the hub module's weights. We simply have to retrieve the model using the correct signature for inferencing.

**Resource Recommendations**

For a comprehensive understanding of Keras model saving and loading, consult the official Keras documentation. The TensorFlow documentation, especially the sections related to TensorFlow Hub, and `tf.saved_model` can provide further insights into managing external modules. Reading examples related to similar problems on community forums is also helpful. I have learned a lot by reading other people's issues when they are saving similar models.
