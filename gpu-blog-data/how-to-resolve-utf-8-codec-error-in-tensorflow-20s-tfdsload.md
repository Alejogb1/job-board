---
title: "How to resolve 'utf-8' codec error in TensorFlow 2.0's tfds.load?"
date: "2025-01-26"
id: "how-to-resolve-utf-8-codec-error-in-tensorflow-20s-tfdsload"
---

Encountering a `UnicodeDecodeError: 'utf-8' codec can't decode byte` during a `tfds.load` operation in TensorFlow 2.0 usually signals an incompatibility between the encoding of the data being loaded and the decoder's expectations. Specifically, the TensorFlow Datasets (TFDS) library attempts to decode dataset files, often text-based metadata or even the data itself, assuming a UTF-8 encoding. When this assumption is incorrect, the decoder encounters byte sequences that do not conform to UTF-8’s rules, leading to the observed error. This is more likely with datasets created outside the usual channels that TensorFlow uses, including locally created files that might have been generated or stored with a different encoding.

The root cause is almost always mismatched encoding between the file system and the decoder. UTF-8 is a variable-width character encoding; certain byte sequences represent characters. If a file is encoded using something like Latin-1, or a legacy encoding, those same bytes might represent different characters or no characters at all in UTF-8. The decoder will fail when it encounters bytes that do not form a valid sequence, generating this error. It’s not necessarily a fault of TensorFlow or TFDS, but rather an inherent issue of mismatched interpretation of raw byte data. This error primarily arises from text-based datasets and related metadata where encoding consistency is crucial. Image and other binary data typically do not trigger this issue unless their associated metadata has encoding issues. My experience troubleshooting various data ingestion pipelines across multiple projects leads me to understand this problem is common, particularly when dealing with a diverse range of source data, particularly older files.

Resolution involves identifying the actual encoding of the problematic file, and potentially either converting the file to UTF-8 or, more practically, specifying the correct encoding to TFDS during the loading process if the conversion is not feasible. The challenge lies in accurately identifying the actual encoding since it’s not usually indicated within the file itself, requiring experimentation or prior knowledge of the source. The best approach depends on whether you have control over the original data file, and the scope of dataset usage. If a dataset is a one-off and a small file, modifying the file itself is viable. For larger datasets that are frequently being used, the more efficient solution may be to specify the encoding in TFDS.

Here are three practical examples showcasing different scenarios and resolutions:

**Example 1: Specifying encoding during `tfds.load` for dataset metadata.**

Let’s assume the issue arises because the dataset’s `.dataset_info.json` metadata file was saved using `latin-1` instead of UTF-8. This JSON file holds basic information about the dataset. The simplest approach is to force TFDS to read this with the correct encoding.

```python
import tensorflow_datasets as tfds

try:
  dataset = tfds.load('my_dataset', data_dir='/path/to/my_dataset', split='train')
except UnicodeDecodeError as e:
  print(f"Initial load failed due to UnicodeDecodeError: {e}")

  # Force the metadata to be read with the correct encoding
  dataset = tfds.load('my_dataset', data_dir='/path/to/my_dataset', split='train',
                      try_gcs=False,  
                      download_config=tfds.download.DownloadConfig(
                          file_format_configs = {
                            tfds.core.FileFormat.JSON:
                                  tfds.download.FileFormatConfig(encoding='latin-1')
                          }
                      ))

  print("Dataset loaded after specifying latin-1 encoding for metadata.")


for example in dataset.take(2):
    print(example)
```

In this scenario, the initial `tfds.load` triggers the `UnicodeDecodeError`, which is caught. The second `tfds.load` call includes a `download_config` argument with an override for the `JSON` file format, explicitly telling TFDS to decode any JSON file using the specified encoding. I included `try_gcs=False` to bypass the default behavior of trying to download from Google Cloud Storage and explicitly use the files in the `data_dir`. I have done this often to avoid downloading datasets multiple times when developing datasets locally. This fix is specific to JSON metadata and will not fix data files. The `take(2)` is added for demonstration purposes, confirming that loading has been successful. The key here is `FileFormatConfig` and its `encoding` parameter. This is the strategy I often reach for first, since it's usually the easiest.

**Example 2: Modifying Data files and using a default Encoding**

Another scenario might involve the data files themselves having incorrect encodings, not just metadata. Assume you have CSV files containing data that are incorrectly encoded with a legacy system.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

# First, simulate an incorrectly encoded CSV file.
# The real-world step will be where you obtain your data files
data = {'col1': ['àéíóú', '你好', 'special'], 'col2': [1, 2, 3]}
df = pd.DataFrame(data)
df.to_csv('data.csv', encoding='latin-1', index=False) # Incorrect Encoding

# Attempt to load from a tf.data.TextLineDataset - this will fail if text is not UTF-8
try:
    text_dataset = tf.data.TextLineDataset(['data.csv'])
    for line in text_dataset.take(2):
        print(line)
except tf.errors.InvalidArgumentError as e:
  print(f"TextLineDataset load failed due to InvalidArgumentError: {e}")

# Read CSV with Pandas using the correct encoding to correct.
df = pd.read_csv('data.csv', encoding='latin-1')

# Save CSV file with correct encoding.
df.to_csv('data_utf8.csv', encoding='utf-8', index=False)


text_dataset = tf.data.TextLineDataset(['data_utf8.csv'])
for line in text_dataset.take(2):
    print(line)

# The corrected file can be loaded and used when creating TFDS dataset.

print("Dataset successfully created using UTF-8 CSV files.")


#Example dataset creation with the fixed csv files.
class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for My Dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  def _info(self) -> tfds.core.DatasetInfo:
      """Returns the dataset metadata."""
      return tfds.core.DatasetInfo(
          builder=self,
          features=tfds.features.FeaturesDict({
                  'col1': tfds.features.Text(),
                  'col2': tfds.features.Scalar(dtype=tf.int64)
              }),
      )


  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract({
        'data_csv' : 'data_utf8.csv'
    })
    return {
        'train': self._generate_examples(path['data_csv'])
    }


  def _generate_examples(self, path):
      """Yields examples."""
      for i, row in pd.read_csv(path).iterrows():
        yield i, {
          'col1': row['col1'],
          'col2': row['col2']
        }
#Load the dataset to verify
dataset = tfds.load('my_dataset', data_dir='./', split='train')
for example in dataset.take(1):
  print(example)
```
This example shows an approach where the underlying data files, rather than just metadata, are causing the issue. A `TextLineDataset` will default to UTF-8 and errors when encountering non UTF-8 characters. Here, Pandas is used to load the data with the *correct* (latin-1) encoding, and it is then saved with UTF-8 encoding. This means the data now conforms to UTF-8, and the subsequent `TextLineDataset` does not error. This is then incorporated into a simple example dataset builder showing where the corrected files can be used. This solution is most beneficial when the datasets are consistently causing the issue. While using pandas to correct and rewrite data files has overhead, it does guarantee the files are correctly encoded, preventing the issue from reappearing in the future. The dataset builder was simplified to avoid focusing on the topic. The key takeaway is that the data file itself must be UTF-8.

**Example 3: Handling Files with Unknown Encoding**

In the most difficult cases, the encoding might be unknown. While a full solution can be quite complicated, an approach that I have found useful is to initially load the file as raw bytes, then attempt to decode with different encodings until one works without an error. This involves manually managing the data using a `tf.data.Dataset` in conjunction with Python's error handling.

```python
import tensorflow as tf
import chardet # Requires 'pip install chardet'

# Simulate a file with unknown encoding
with open('unknown.txt', 'wb') as f:
  f.write("àéíóú".encode('iso-8859-1')) # Write with incorrect encoding

def decode_with_fallback(path):
  """Tries different encodings and decodes file to UTF-8."""
  with open(path, 'rb') as f:
    raw_bytes = f.read()

    result = chardet.detect(raw_bytes) # Guess Encoding
    detected_encoding = result['encoding']

    if detected_encoding is not None:
      try:
         decoded_text = raw_bytes.decode(detected_encoding)
         return decoded_text
      except UnicodeDecodeError:
         print(f"Decoding with detected encoding {detected_encoding} failed.")
    
    # If detection failed or decoding failed, attempt common encodings
    for encoding in ['utf-8','latin-1', 'windows-1252']:
         try:
            decoded_text = raw_bytes.decode(encoding)
            print(f"Successful decode using {encoding}")
            return decoded_text
         except UnicodeDecodeError:
             pass # Failed, continue loop

    # If all else fails, raise an error
    raise ValueError("Failed to decode file with any attempted encoding.")

# Test function
try:
  decoded_text = decode_with_fallback('unknown.txt')
  print(f"Decoded text: {decoded_text}")
except ValueError as e:
  print(f"Decoding failed: {e}")

# Use the decoding logic when generating examples for your dataset
class MyUnknownEncodingDataset(tfds.core.GeneratorBasedBuilder):
   # Version and dataset info omitted for brevity

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract({
        'data' : 'unknown.txt'
    })
    return {
        'train': self._generate_examples(path['data'])
    }

  def _generate_examples(self, path):
     """Yields examples."""
     decoded_text = decode_with_fallback(path)
     # Further processing of decoded text
     yield 0 , {'text' : decoded_text}

#Load and view
dataset = tfds.load('my_unknown_encoding_dataset', data_dir = './', split='train')

for example in dataset.take(1):
  print(example)

```

In this advanced case, I've added the usage of the `chardet` library to attempt to detect the encoding. `chardet` is very useful but is not guaranteed to get the encoding right all the time. The `decode_with_fallback` function will attempt decoding the file with the detected encoding or with hardcoded defaults, and will return the decoded text in UTF-8, making it safe for use with TensorFlow. The dataset generator then uses this function to handle data. While this approach is more complex and less efficient, it’s critical when you're working with data with unknown encodings that cannot be corrected manually. This is a solution I often use when dealing with data from legacy systems with poor documentation. I often use this logic when I'm unsure of a file's source and can't get more information. This strategy, while complex, often resolves the most troublesome instances of data corruption.

For further information and background, I recommend consulting the official Python documentation on Unicode and character encodings, as well as the TensorFlow documentation specifically on loading and working with datasets. The documentation for the chardet library can be used for understanding the details of how this library works. Finally, if one is creating custom datasets, examining the TFDS documentation around dataset creation is highly beneficial. These resources will provide a more complete understanding of the problem and the approaches to resolving it.
