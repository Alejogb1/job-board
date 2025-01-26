---
title: "How do I download a custom dataset for a Google Custom Model Maker TensorFlow model?"
date: "2025-01-26"
id: "how-do-i-download-a-custom-dataset-for-a-google-custom-model-maker-tensorflow-model"
---

The primary challenge in utilizing custom datasets with Google Custom Model Maker lies in formatting the data into a structure TensorFlow can readily ingest, particularly when scaling beyond small, toy examples. I've personally encountered this during several projects, initially struggling with resource bottlenecks and inefficient data loading, which significantly hampered training speed. The approach varies depending on data type (images, text, audio, etc.) and whether you're using a hosted Colab environment or local setup.

The process broadly consists of three key stages: data preparation, data loading, and integration with the Model Maker API. Data preparation involves organizing your dataset into a structure that is understood by TensorFlow's data pipelines, typically via `tf.data.Dataset`. Loading then utilizes TensorFlow's API to create a dataset object from your prepared data. Finally, the Model Maker's API facilitates the use of this dataset for training and evaluation.

For image classification models, a common scenario, data should ideally be structured in a directory hierarchy where each subdirectory represents a class, and each file within a subdirectory is an image belonging to that class. This organization mirrors the expected input structure of `tf.keras.utils.image_dataset_from_directory`. If your data is in another format, such as a single archive containing all images, a custom preparation script would be necessary to reorganize it.

Here's an example showcasing how to download a custom dataset from Google Cloud Storage (GCS), re-organize it, and create a TensorFlow dataset for image classification. Assume a GCS bucket holds a ZIP archive named `my_dataset.zip`, containing images spread across directories corresponding to classes.

```python
import tensorflow as tf
import shutil
import os
import zipfile

# 1. Define paths and download settings
gcs_bucket_path = "gs://your-bucket-name/my_dataset.zip" #Replace with actual GCS path
local_download_path = "my_dataset.zip"
extracted_dataset_path = "my_dataset"
image_size = (224, 224) # Example image size for model training
batch_size = 32 # Batch size for training

# 2. Download and extract the dataset
try:
    tf.io.gfile.copy(gcs_bucket_path, local_download_path)
    with zipfile.ZipFile(local_download_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dataset_path)
    os.remove(local_download_path) # Clean up the downloaded zip

    print("Dataset downloaded and extracted successfully!")

except Exception as e:
    print(f"Error downloading or extracting dataset: {e}")
    exit()

# 3. Create TensorFlow dataset from the directory structure
train_dataset = tf.keras.utils.image_dataset_from_directory(
    extracted_dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    extracted_dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
)

print("TensorFlow datasets created successfully!")

# Example: Accessing dataset shape
for images, labels in train_dataset.take(1):
  print("Shape of batch of images:", images.shape)
  print("Shape of labels:", labels.shape)

# The train_dataset and validation_dataset are ready for use in Model Maker
```

In this example, the code downloads a zip archive, extracts its content into a structured directory, then employs `tf.keras.utils.image_dataset_from_directory` to generate training and validation `tf.data.Dataset` objects. The `validation_split` parameter automatically separates the data. The `seed` ensures reproducible splitting. Batch size is configurable, and the example prints the shape of one batch to confirm correct dataset creation. The extracted folder's structure must conform to the `image_dataset_from_directory` expectations: `extracted_dataset_path/class_name/image.jpg`. Any deviation will lead to errors during dataset creation.

The process for textual data requires different functions, focusing on converting text to numerical representations. The following demonstrates preparing a text classification dataset from a CSV file. I've used this approach frequently for sentiment analysis.

```python
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Define paths and settings
csv_path = 'my_text_data.csv'  # Path to your CSV file
text_column = 'text' # The name of the column containing the text
label_column = 'label' # Name of column containing labels
batch_size = 32
max_features = 5000 # Maximum vocabulary size
sequence_length = 128 # Maximum sequence length

# 2. Read data using pandas
try:
    df = pd.read_csv(csv_path)
    print("CSV data loaded successfully.")

except Exception as e:
    print(f"Error loading csv file: {e}")
    exit()

# 3. Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Convert labels to numpy arrays
train_labels = np.array(train_df[label_column])
val_labels = np.array(val_df[label_column])


# 5. Create a Text Vectorization Layer
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length
    )

# Adapt vectorizer with training texts
vectorizer.adapt(train_df[text_column])

# 6. Vectorize text data using map function
def vectorize_text(text,label):
    return vectorizer(text), label

train_dataset = tf.data.Dataset.from_tensor_slices((list(train_df[text_column]),train_labels))
train_dataset = train_dataset.map(vectorize_text)
train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((list(val_df[text_column]),val_labels))
val_dataset = val_dataset.map(vectorize_text)
val_dataset = val_dataset.batch(batch_size)

# Example: Print batch shape
for texts, labels in train_dataset.take(1):
  print("Shape of batch of text:", texts.shape)
  print("Shape of labels:", labels.shape)


print("Text datasets created successfully!")

# train_dataset and val_dataset are ready for Model Maker
```

This code reads a CSV file, splits it into training and validation sets, then converts text data into sequences of numerical tokens utilizing `tf.keras.layers.TextVectorization`. This layer is adapted to the training data's vocabulary, allowing for out-of-vocabulary words to be handled. Both training and validation data are then transformed into `tf.data.Dataset` objects with batched numerical representations.  The CSV structure expects at least two columns, `text` and `label`, but this is configurable to the column names as declared in the code.

Finally, the following example illustrates downloading data from Hugging Face datasets, which I often use for evaluating model performance on benchmark datasets.

```python
import tensorflow as tf
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
import numpy as np

# 1. Define dataset name and settings
dataset_name = 'glue'
subset = 'sst2' # Example dataset
batch_size = 32
max_seq_length = 128 # Max sequence length
tokenizer_name = 'bert-base-uncased' # Tokenizer for pre-processing


# 2. Download dataset from HuggingFace
try:
  raw_datasets:DatasetDict = load_dataset(dataset_name, subset)
  print("Hugging Face dataset loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face dataset: {e}")
    exit()


# 3. Load the tokenizer
try:
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print("Tokenizer Loaded successfully.")
except ImportError:
  print("transformers library not installed, you can install it by using command: pip install transformers")
  exit()

# 4. Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length = max_seq_length)

# 5. Apply tokenization
tokenized_datasets:DatasetDict = raw_datasets.map(tokenize_function, batched=True)

# 6. Create numpy data for TF
train_numpy_data = {key: np.array(value) for key,value in tokenized_datasets["train"].items() if key != "idx"}
val_numpy_data = {key: np.array(value) for key,value in tokenized_datasets["validation"].items() if key != "idx"}

# 7. Convert to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_numpy_data),np.array(tokenized_datasets["train"]["label"])))
train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_numpy_data),np.array(tokenized_datasets["validation"]["label"])))
val_dataset = val_dataset.batch(batch_size)


#Example of accessing shapes
for texts, labels in train_dataset.take(1):
  print("Shape of batch of text:", texts["input_ids"].shape)
  print("Shape of labels:", labels.shape)

print("TensorFlow Datasets created successfully.")

# The train_dataset and val_dataset are ready for use with Model Maker
```

This code leverages the `datasets` library from Hugging Face to download a benchmark dataset, then employs a tokenizer, specifically `BertTokenizerFast`, to transform the text. It then converts this tokenized data into `tf.data.Dataset` for TensorFlow consumption. The data is formatted as a dictionary of tensor slices to handle multi-input scenarios with BERT style models and the labels are processed as an individual tensor for label inputs. The example prints the shape of a batch to confirm correct loading. It's important to note that Hugging Face dataset structures can vary greatly, so adjustments may be needed for different datasets.

For further information, refer to the TensorFlow documentation for `tf.data.Dataset`, specifically the sections dealing with loading data from various sources such as files, directories, and Python generators. In addition, exploring the documentation of the specific model maker you wish to use is essential, as there may be specific expectations about the structure and content of the dataset. Finally, researching tutorials on data pipeline implementations with `tf.data.Dataset` can improve understanding of efficient data processing techniques. These resources have provided me with the necessary foundation to build diverse and effective models.
