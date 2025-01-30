---
title: "Can FastAI's `get_preds()` handle dictionary datatypes?"
date: "2025-01-30"
id: "can-fastais-getpreds-handle-dictionary-datatypes"
---
FastAI's `get_preds()` method, in its standard implementation, does not directly support dictionary datatypes as input.  My experience working on several image classification and natural language processing projects using FastAI, primarily versions 2 and 1, revealed this limitation consistently.  The function anticipates a structured data input, typically a PyTorch `DataLoader` output or a tensor representing a batch of data points.  Dictionaries, being unordered collections of key-value pairs, lack the inherent structure needed for efficient batch processing within the FastAI framework.  This necessitates a pre-processing step to transform dictionary data into a suitable format before using `get_preds()`.


The core issue stems from `get_preds()`'s dependence on the underlying model architecture and its expectation of input tensors with specific dimensions.  A dictionary, without prior transformation, cannot easily be mapped to these input expectations.  The model's forward pass requires correctly shaped tensors representing features, and a dictionary inherently fails to provide this inherent structure. Consequently, attempting to pass a dictionary directly will result in a `TypeError` or a more cryptic error message related to tensor shape mismatch.


The solution involves transforming the dictionary data into a suitable format, typically a NumPy array or a PyTorch tensor.  The precise transformation will depend on the structure and contents of the dictionary itself, particularly how features are encoded within the dictionary's key-value pairs.  This transformation must ensure the data is organized appropriately for batch processing.


Let's illustrate this with three code examples showcasing different dictionary structures and the necessary transformation steps before using `get_preds()`.


**Example 1: Dictionary with Numerical Features**

Consider a dictionary where each key represents a data point, and the value is another dictionary containing numerical features.  This setup mirrors a common scenario in tabular data where each row corresponds to an observation and the columns are features.

```python
import numpy as np
from fastai.vision.all import *

# Sample data – representing features for three data points
data_dict = {
    'point1': {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0},
    'point2': {'feature1': 4.0, 'feature2': 5.0, 'feature3': 6.0},
    'point3': {'feature1': 7.0, 'feature2': 8.0, 'feature3': 9.0}
}

# Transform the dictionary into a NumPy array
features = []
for key, value in data_dict.items():
    features.append([value['feature1'], value['feature2'], value['feature3']])
features_array = np.array(features)

# Assuming 'learn' is your trained FastAI learner object
# and the model expects input shape (batch_size, 3)
preds = learn.get_preds(dl=DataLoader(Tensor(features_array))) # using a DataLoader for correct handling
print(preds[0]) # Accessing the predictions
```

In this example, the nested dictionaries are flattened into a NumPy array.  The `DataLoader` is crucial as it handles batching and ensures compatibility with FastAI's internal workings.  Directly passing `Tensor(features_array)` might work for small datasets but is generally not recommended for efficiency and robustness.


**Example 2: Dictionary with Textual Features**

Handling textual features necessitates a different approach. We'll assume a dictionary where keys represent data points, and the values are strings representing textual data.

```python
import numpy as np
from fastai.text.all import *

# Sample data - three sentences
text_dict = {
    'doc1': 'This is a sentence.',
    'doc2': 'Another sentence here.',
    'doc3': 'And yet another one.'
}

# Assuming 'tokenizer' is a pre-trained tokenizer and 'vocab' is the vocabulary.
tokens = [tokenizer(text) for text in text_dict.values()]
numericalized_tokens = [[vocab[token] for token in doc] for doc in tokens]
padded_tokens = pad_sequence(numericalized_tokens, padding_value=vocab['<pad>']) # handle varying sequence lengths

# Assuming learn is your text classifier learner.
tensor_data = Tensor(padded_tokens)
preds = learn.get_preds(dl=DataLoader(tensor_data))
print(preds[0])
```

Here, we utilize a pre-trained tokenizer and vocabulary to convert the textual data into numerical representations.  The `pad_sequence` function is critical for handling sequences of varying lengths, a common occurrence with text data.  Padding ensures all sequences have the same length, which is a prerequisite for batch processing by PyTorch and FastAI.


**Example 3:  Dictionary with Mixed Data Types**

This example demonstrates a more complex scenario with a mix of numerical and categorical features.

```python
import numpy as np
from fastai.tabular.all import *

mixed_data = {
    'point1': {'numeric_feature': 10, 'categorical_feature': 'A'},
    'point2': {'numeric_feature': 20, 'categorical_feature': 'B'},
    'point3': {'numeric_feature': 30, 'categorical_feature': 'A'}
}

# Create a DataFrame for easier processing
df = pd.DataFrame.from_dict(mixed_data, orient='index')

# Process categorical features - One hot encoding for illustration
df['categorical_feature'] = pd.Categorical(df['categorical_feature'])
df = pd.get_dummies(df, columns=['categorical_feature'], prefix=['categorical'])

# Convert to NumPy array
features_array = df.values

# Assuming learn is your tabular learner
preds = learn.get_preds(dl=DataLoader(Tensor(features_array)))
print(preds[0])
```

Here, we leverage Pandas for efficient data manipulation.  Categorical features are one-hot encoded before conversion to a NumPy array.  This approach allows for effective handling of mixed data types, a common reality in many machine learning applications.



In conclusion, while FastAI's `get_preds()` doesn't directly handle dictionaries, with careful pre-processing and transformation into appropriate tensor formats, along with utilizing DataLoaders for efficient batch handling, predictions can be obtained from models trained within the FastAI ecosystem.  The choice of transformation methodology depends heavily on the specifics of the dictionary structure and the data types within.   Remember to consult the FastAI documentation and relevant PyTorch tutorials for further insights into tensor manipulation and data loading best practices.  Furthermore, exploring advanced techniques like custom data loaders for exceptionally complex dictionary structures may prove beneficial.
