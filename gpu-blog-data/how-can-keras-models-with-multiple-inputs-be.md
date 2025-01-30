---
title: "How can Keras models with multiple inputs be constructed using custom data generators?"
date: "2025-01-30"
id: "how-can-keras-models-with-multiple-inputs-be"
---
Implementing custom data generators for Keras models with multiple inputs requires careful handling of data organization and batching, particularly when dealing with complex datasets or resource limitations. I've encountered this specific challenge multiple times during my tenure developing machine learning models for sensor data analysis. The key lies in ensuring that the generator yields data in a format Keras expects for multi-input models— specifically, a tuple or list where each element corresponds to an input layer.

The core issue is that typical generators often output a single NumPy array or tensor representing all features. With multiple input layers, we need to restructure our data into a format that matches the input layer definition of our model. This involves not just batching, but also segmenting the batched data. For instance, if a model accepts an image and a time series, the generator has to produce batches where the first element is a batch of images, and the second is a batch of corresponding time series data.

To achieve this, we define a custom Python generator function. This function should read your raw data (which might be spread across multiple files or databases), organize it into the required input groups, apply any necessary preprocessing steps, and then yield batches. Here’s how that pattern looks in practice:

**Explanation of the Process**

1. **Data Loading:** The generator starts by loading the necessary data, potentially iterating through files or using more complex data access methods. The key here is to maintain a consistent mapping between the different input streams. If your data is stored in different formats, you’ll handle conversion or parsing at this point.
2. **Preprocessing:** Raw data is rarely suitable for models. The generator performs transformations like normalization, resizing, or feature extraction before the batch is created. This step must be applied consistently across all corresponding elements within the same batch. For example, applying image normalization to image data only.
3. **Batch Construction:** The function then gathers elements to create batches of appropriate size. The generator creates a tuple of batches, where each element of the tuple is a batch that will be fed into the corresponding model input layer.
4. **Yielding:** The crucial step is to `yield` a tuple where the first element corresponds to the first input layer, the second to the second, and so on. It's imperative that the order matches your Keras model definition. The generator also typically yields the targets (the `y` values), which may similarly be a single batch, or a tuple in the case of multiple outputs.

**Code Examples**

Here are three code examples that illustrate this process with increasing levels of complexity:

**Example 1: Simple two-input model**

This example deals with a simple model that accepts two numeric inputs. The data is represented by two numpy arrays `input1_data` and `input2_data`, and we will generate batches of size 32.

```python
import numpy as np
import tensorflow as tf

def multi_input_generator_simple(input1_data, input2_data, batch_size):
  num_samples = input1_data.shape[0]
  indices = np.arange(num_samples)
  np.random.shuffle(indices) # optional: shuffle for stochastic training
  i = 0
  while True:
      batch_indices = indices[i * batch_size:(i+1)*batch_size]
      if len(batch_indices) < batch_size:
        # If we reached end, reset the index.
        i = 0
        batch_indices = indices[i * batch_size:(i+1)*batch_size]
        
      batch_input1 = input1_data[batch_indices]
      batch_input2 = input2_data[batch_indices]

      # Example target (replace with actual labels)
      batch_y = np.random.rand(len(batch_indices), 1)

      yield (batch_input1, batch_input2), batch_y
      i += 1

# Example Usage
input1_data = np.random.rand(1000, 10)
input2_data = np.random.rand(1000, 20)
batch_size = 32
generator = multi_input_generator_simple(input1_data, input2_data, batch_size)

# Example model to test generator
input1 = tf.keras.Input(shape=(10,))
input2 = tf.keras.Input(shape=(20,))

merged = tf.keras.layers.concatenate([input1, input2])
output = tf.keras.layers.Dense(1)(merged)
model = tf.keras.Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer="adam", loss="mse")

model.fit(generator, steps_per_epoch=len(input1_data)//batch_size, epochs=2)
```
**Commentary:** This simple example demonstrates the basic structure of the generator. The essential parts are the shuffling of the indices, creation of batches for each input, and yielding the batches as a tuple. The example includes a basic model, and how you can use fit to train with the data from the generator. The key is to return data corresponding to the input order in your model.

**Example 2: Image and Text Data Generator**

This example demonstrates a generator for a model that accepts image data (assuming all images are pre-loaded into memory for simplicity) and accompanying text data (tokenized sequences).

```python
import numpy as np
import tensorflow as tf

def image_text_generator(image_data, text_data, batch_size, max_text_len):
    num_samples = image_data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    i = 0
    while True:
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        if len(batch_indices) < batch_size:
            i = 0
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        batch_images = image_data[batch_indices]
        batch_texts = text_data[batch_indices]

        # Ensure text sequences have the same length by padding
        batch_texts_padded = tf.keras.preprocessing.sequence.pad_sequences(batch_texts, maxlen=max_text_len, padding='post')

        batch_y = np.random.rand(len(batch_indices), 1)
        yield (batch_images, batch_texts_padded), batch_y
        i+=1

# Example Usage
image_data = np.random.rand(1000, 64, 64, 3)
text_data = [np.random.randint(0, 1000, np.random.randint(5, 20)) for _ in range(1000)]
batch_size = 32
max_text_len = 20
generator = image_text_generator(image_data, text_data, batch_size, max_text_len)

# Example model to test generator
input_img = tf.keras.Input(shape=(64, 64, 3))
input_txt = tf.keras.Input(shape=(max_text_len,))

conv = tf.keras.layers.Conv2D(32, (3,3))(input_img)
flat = tf.keras.layers.Flatten()(conv)
emb = tf.keras.layers.Embedding(input_dim=1000, output_dim=32)(input_txt)
lstm = tf.keras.layers.LSTM(32)(emb)
merged = tf.keras.layers.concatenate([flat, lstm])
output = tf.keras.layers.Dense(1)(merged)

model = tf.keras.Model(inputs=[input_img, input_txt], outputs=output)
model.compile(optimizer='adam', loss='mse')

model.fit(generator, steps_per_epoch=len(image_data)//batch_size, epochs=2)
```

**Commentary:** This example introduces a more realistic scenario involving images and variable-length text. The key here is the use of `tf.keras.preprocessing.sequence.pad_sequences` to handle text sequence length inconsistencies within a batch. Each batch needs consistent shapes for the model to process correctly. Additionally, this highlights the need to include the correct text embedding layer, which corresponds to the vocabulary of the input data. The example model shows a potential usage with convolutional layers processing images, and recurrent layers (LSTM) working with the text embeddings.

**Example 3: Multi-File Data Generator with Preprocessing**

This example simulates a more complex data situation where sensor readings and associated metadata come from separate files. We simulate this by creating some mock data files.

```python
import numpy as np
import tensorflow as tf
import os

def create_mock_data(num_samples, data_dir):
    for i in range(num_samples):
        sensor_data = np.random.rand(100,5)
        metadata = {'location': f'loc_{i%3}', 'time': i*100}
        np.save(os.path.join(data_dir, f'sensor_{i}.npy'), sensor_data)
        with open(os.path.join(data_dir, f'meta_{i}.txt'), 'w') as f:
            f.write(str(metadata))

def multi_file_generator(data_dir, batch_size):
    filenames = [f.split('.')[0].split('_')[-1] for f in os.listdir(data_dir) if f.endswith('.npy')]
    filenames.sort(key=int)
    num_samples = len(filenames)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    i = 0
    while True:
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        if len(batch_indices) < batch_size:
            i = 0
            batch_indices = indices[i*batch_size:(i+1)*batch_size]

        batch_sensor_data = []
        batch_metadata_loc = [] # we extract the location
        for idx in batch_indices:
            sensor_file = os.path.join(data_dir, f'sensor_{filenames[idx]}.npy')
            meta_file = os.path.join(data_dir, f'meta_{filenames[idx]}.txt')

            sensor_data = np.load(sensor_file)
            with open(meta_file, 'r') as f:
                meta_data = eval(f.read()) # caution: using eval
            batch_sensor_data.append(sensor_data)
            batch_metadata_loc.append(meta_data['location'])
        
        batch_sensor_data = np.array(batch_sensor_data)
        batch_metadata_loc = np.array(batch_metadata_loc)
        # convert to numerical data for embedding layer
        location_to_id = {'loc_0':0, 'loc_1':1, 'loc_2':2}
        batch_metadata_loc_ids = np.vectorize(location_to_id.get)(batch_metadata_loc)

        batch_y = np.random.rand(len(batch_indices), 1)
        yield (batch_sensor_data, batch_metadata_loc_ids), batch_y
        i += 1

# Example Usage
data_dir = "mock_data"
os.makedirs(data_dir, exist_ok=True)
num_samples = 1000
batch_size = 32

create_mock_data(num_samples, data_dir)

generator = multi_file_generator(data_dir, batch_size)

# Example model to test generator
input_sensor = tf.keras.Input(shape=(100, 5))
input_meta = tf.keras.Input(shape=(1,))
conv = tf.keras.layers.Conv1D(filters=32, kernel_size=3)(input_sensor)
flat = tf.keras.layers.Flatten()(conv)

emb = tf.keras.layers.Embedding(input_dim=3, output_dim=32)(input_meta)
emb_flat = tf.keras.layers.Flatten()(emb)

merged = tf.keras.layers.concatenate([flat, emb_flat])
output = tf.keras.layers.Dense(1)(merged)

model = tf.keras.Model(inputs=[input_sensor, input_meta], outputs=output)
model.compile(optimizer='adam', loss='mse')

model.fit(generator, steps_per_epoch=len(os.listdir(data_dir))//2//batch_size, epochs=2)
```
**Commentary:** This more realistic example shows a file-based data loader, where data is loaded from disk. This example also highlights preprocessing data loaded from files, ensuring consistent shape and format. This demonstrates a more real world example where different data sources, and corresponding preprocessing, can be handled. The `eval` call is included here to demonstrate a common (but possibly risky) method to handle reading simple python dicts from files. In a production environment, using other methods, such as json would be a more robust approach.

**Resource Recommendations:**

To further understand custom data generators in Keras, I recommend consulting the official TensorFlow documentation, which provides detailed information on `tf.data.Dataset` and `keras.utils.Sequence` class, both useful tools in handling larger or more complex datasets, especially when working with multiple input layers. For advanced use cases involving multiple inputs or complex data types, consider researching different techniques for data parallelization and efficient batching strategies, typically discussed in academic publications on distributed training or high-performance computing. Additionally, practical implementation details are often demonstrated on open-source projects and online tutorials focusing on specific model architectures or data analysis tasks.
