---
title: "How can I train a multi-input TensorFlow Keras model using a single dataset?"
date: "2025-01-30"
id: "how-can-i-train-a-multi-input-tensorflow-keras"
---
The challenge of training a Keras model with multiple inputs using a single dataset often stems from the need to represent distinct data modalities, such as text and numerical features, which require specific preprocessing pipelines. My experience working with personalized recommendation systems highlighted this exact issue; we had user text profiles alongside their purchase histories, both needed as inputs to a model predicting future interactions, yet residing within the same user database table. The key lies in correctly preparing the dataset and defining the Keras model to consume these disparate input streams.

Essentially, we are not truly creating "multiple datasets," but rather extracting different feature sets from our single dataset. This single dataset, typically stored in a tabular format (like a Pandas DataFrame, or a CSV file), holds all necessary information. The core concept revolves around preprocessing this single data source to generate input arrays which correspond to the different input layers in the Keras model. These arrays will be fed into their respective layers via the functional API in Keras. This separation is not at the storage level; rather, it occurs in the data loading and preprocessing stage. The single dataset acts as the source for each of the inputs in the final model.

To illustrate, consider a scenario where we have user data containing both categorical features (like location or membership type) and numerical features (such as age or income). In Keras, we would have distinct input layers for each type of feature. The categorical data might be passed through an Embedding layer, while the numerical data might undergo some normalization before feeding into a dense layer. Before this, the data processing pipeline must prepare the correct inputs for each of these layers.

Let's examine some code to demonstrate this. Assume the dataset is stored as a Pandas DataFrame named `df`. For this example, I will create a simplified, conceptual DataFrame for illustration. Note that in real-world scenarios, the DataFrame might be read from a file using `pd.read_csv()` or similar.

**Code Example 1: Basic Categorical and Numerical Inputs**

```python
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample DataFrame (replace with actual dataset loading)
data = {'location': ['New York', 'London', 'Tokyo', 'New York', 'London'],
        'membership_type': ['premium', 'basic', 'premium', 'basic', 'premium'],
        'age': [25, 30, 40, 22, 35],
        'income': [60000, 75000, 120000, 55000, 90000],
        'target': [0, 1, 1, 0, 1]}
df = pd.DataFrame(data)

# 1. Data Preparation
categorical_features = ['location', 'membership_type']
numerical_features = ['age', 'income']
target_feature = 'target'

# One-hot encode categorical features
encoded_cat_features = pd.get_dummies(df[categorical_features])

# Scale numerical features
scaler = StandardScaler()
scaled_num_features = scaler.fit_transform(df[numerical_features])
scaled_num_features = pd.DataFrame(scaled_num_features, columns=numerical_features)

# Combine preprocessed features
X = pd.concat([encoded_cat_features, scaled_num_features], axis=1)
y = df[target_feature]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Separate inputs for the model
categorical_input_train = X_train.iloc[:, :len(encoded_cat_features.columns)]
numerical_input_train = X_train.iloc[:, len(encoded_cat_features.columns):]
categorical_input_test = X_test.iloc[:, :len(encoded_cat_features.columns)]
numerical_input_test = X_test.iloc[:, len(encoded_cat_features.columns):]


# 2. Model Definition
cat_input = keras.Input(shape=(len(encoded_cat_features.columns),), name='categorical_input')
num_input = keras.Input(shape=(len(numerical_features),), name='numerical_input')

cat_dense = keras.layers.Dense(64, activation='relu')(cat_input)
num_dense = keras.layers.Dense(32, activation='relu')(num_input)

merged = keras.layers.concatenate([cat_dense, num_dense])
output = keras.layers.Dense(1, activation='sigmoid')(merged)


model = keras.Model(inputs=[cat_input, num_input], outputs=output)

# 3. Model Compilation and Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=[categorical_input_train, numerical_input_train], y=y_train, epochs=10)

_, accuracy = model.evaluate(x=[categorical_input_test, numerical_input_test], y=y_test)
print(f"Test Accuracy: {accuracy}")
```

**Commentary for Code Example 1:**
Here, the data is first processed by one-hot encoding categorical variables and scaling numerical ones. The processed features are concatenated, and then divided again to correspond to the categorical and numerical model inputs using pandas slicing. It's crucial to understand that the same transformations are applied to both the training and the testing data. We then define separate `Input` layers in the Keras model, which have different shapes. Note the use of the functional API, which allows us to explicitly create the multiple input architecture. Finally, we pass the separate feature sets during the `fit` and `evaluate` calls.

**Code Example 2: Text and Numerical Inputs**

```python
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample DataFrame (replace with actual dataset loading)
data = {'user_text': ['I love this product', 'it was okay', 'terrible experience', 'amazing result', 'not so great'],
        'purchase_count': [10, 2, 1, 15, 5],
        'target': [1, 0, 0, 1, 0]}
df = pd.DataFrame(data)

# 1. Data Preparation
text_feature = 'user_text'
numerical_feature = 'purchase_count'
target_feature = 'target'

# Text vectorization
max_tokens = 100 # Define the size of your vocabulary
vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=20) #output_sequence length could be an average or max sequence length
vectorizer.adapt(df[text_feature])
text_input = vectorizer(df[text_feature].values)

# Scale numerical feature
scaler = StandardScaler()
numerical_input = scaler.fit_transform(df[[numerical_feature]])
numerical_input= pd.DataFrame(numerical_input, columns=[numerical_feature])

# Split into train and test sets (using index-based separation)
train_indices = np.random.choice(df.index, size=int(len(df) * 0.8), replace=False)
test_indices = df.index.difference(train_indices)

text_input_train = text_input[train_indices]
numerical_input_train = numerical_input.loc[train_indices]
y_train = df[target_feature].loc[train_indices]
text_input_test = text_input[test_indices]
numerical_input_test = numerical_input.loc[test_indices]
y_test = df[target_feature].loc[test_indices]

# 2. Model Definition
text_input_layer = keras.Input(shape=(text_input_train.shape[1],), name='text_input')
num_input_layer = keras.Input(shape=(1,), name='numerical_input')

embedding_layer = keras.layers.Embedding(input_dim=max_tokens, output_dim=16)(text_input_layer)
text_flatten = keras.layers.Flatten()(embedding_layer)

num_dense = keras.layers.Dense(32, activation='relu')(num_input_layer)

merged = keras.layers.concatenate([text_flatten, num_dense])
output = keras.layers.Dense(1, activation='sigmoid')(merged)

model = keras.Model(inputs=[text_input_layer, num_input_layer], outputs=output)


# 3. Model Compilation and Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=[text_input_train, numerical_input_train.values], y=y_train, epochs=10)
_, accuracy = model.evaluate(x=[text_input_test, numerical_input_test.values], y=y_test)
print(f"Test Accuracy: {accuracy}")
```

**Commentary for Code Example 2:**
This example shows the incorporation of text data processed using TextVectorization. This layer performs tokenization and converts the text into sequences of integers which can then be processed by Embedding layer. We maintain the `numerical_input` preparation by standard scaling numerical values. The input data is also separated before feeding it into the model during the training stage. The data split is handled by indexing to ensure that the text and numerical data are separated on the same sets of examples. Note the input shape difference again in `keras.Input`.

**Code Example 3: Handling Image and Numerical Inputs**

```python
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from PIL import Image
import io

# Dummy Image Data Generation (Replace with actual image loading)
def generate_dummy_images(num_images, size=(64, 64)):
    images = []
    for _ in range(num_images):
        img = np.random.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
        img_bytes = io.BytesIO()
        Image.fromarray(img).save(img_bytes, format='PNG')
        images.append(img_bytes.getvalue()) #Image bytes
    return images


# Sample DataFrame (replace with actual dataset loading)
num_images = 5
image_data = generate_dummy_images(num_images)
data = {'image_bytes': image_data,
        'size': [10, 2, 5, 12, 3],
        'target': [1, 0, 0, 1, 1]}

df = pd.DataFrame(data)

# 1. Data Preparation
image_feature = 'image_bytes'
numerical_feature = 'size'
target_feature = 'target'
image_size = (64, 64)

# Decode and preprocess images
decoded_images = [tf.image.decode_png(img_bytes, channels=3) for img_bytes in df[image_feature]] # Decode bytes to images
resized_images = [tf.image.resize(img, image_size) for img in decoded_images] # Resize to a standard size
image_input = tf.stack(resized_images) # Stack into a tensor


# Scale numerical feature
scaler = StandardScaler()
numerical_input = scaler.fit_transform(df[[numerical_feature]])
numerical_input= pd.DataFrame(numerical_input, columns=[numerical_feature])

# Split into train and test sets (using index-based separation)
train_indices = np.random.choice(df.index, size=int(len(df) * 0.8), replace=False)
test_indices = df.index.difference(train_indices)

image_input_train = image_input.numpy()[train_indices]
numerical_input_train = numerical_input.loc[train_indices]
y_train = df[target_feature].loc[train_indices]
image_input_test = image_input.numpy()[test_indices]
numerical_input_test = numerical_input.loc[test_indices]
y_test = df[target_feature].loc[test_indices]

# 2. Model Definition
image_input_layer = keras.Input(shape=(image_size[0], image_size[1], 3), name='image_input')
num_input_layer = keras.Input(shape=(1,), name='numerical_input')

conv_layer = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input_layer)
pool_layer = keras.layers.MaxPooling2D((2,2))(conv_layer)
flatten_layer = keras.layers.Flatten()(pool_layer)


num_dense = keras.layers.Dense(32, activation='relu')(num_input_layer)

merged = keras.layers.concatenate([flatten_layer, num_dense])
output = keras.layers.Dense(1, activation='sigmoid')(merged)


model = keras.Model(inputs=[image_input_layer, num_input_layer], outputs=output)

# 3. Model Compilation and Training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=[image_input_train, numerical_input_train.values], y=y_train, epochs=10)
_, accuracy = model.evaluate(x=[image_input_test, numerical_input_test.values], y=y_test)
print(f"Test Accuracy: {accuracy}")
```

**Commentary for Code Example 3:**
This code demonstrates the combination of image and numerical features. The images are initially stored as byte strings in the DataFrame and then decoded into pixel data which is then stacked into a tensor. Note that the output shape for image input layer has three dimensions(height, width, channels) unlike the other inputs. Convolutional and Pooling layers are introduced to handle spatial image features. Finally, like the previous examples, inputs are separated before passing to `fit` and `evaluate`.

These examples highlight the general strategy. The core idea remains the same: transform your single dataset into multiple input arrays, ensuring each array's shape aligns with the corresponding input layer of the model. Further experimentation with different preprocessing techniques, architectures and more complex data loading pipelines, including using `tf.data.Dataset`, can improve model performance. For comprehensive exploration, I would suggest exploring the official TensorFlow documentation on data loading and preprocessing, as well as books on deep learning using Keras. Furthermore, academic papers that detail deep learning architectures for multimodal learning can offer valuable insights. Finally, online courses focused on deep learning, particularly those emphasizing data handling, can be a great resource. These materials will solidify your understanding and provide the necessary tools to handle these types of situations in diverse real-world projects.
