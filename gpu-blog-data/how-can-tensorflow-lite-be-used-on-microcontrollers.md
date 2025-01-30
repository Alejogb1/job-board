---
title: "How can TensorFlow Lite be used on microcontrollers to predict with missing values and categorical variables?"
date: "2025-01-30"
id: "how-can-tensorflow-lite-be-used-on-microcontrollers"
---
TensorFlow Lite's resource-constrained design enables machine learning inference on microcontrollers, yet handling missing values and categorical variables presents significant challenges given their limited computational capabilities. I've encountered these specific hurdles while deploying predictive maintenance models on embedded systems controlling industrial machinery, where sensor readings frequently include gaps and feature sets often comprise categorical identifiers. Effective microcontroller-based prediction under these conditions necessitates preprocessing at the model creation stage and optimized runtime logic.

Fundamentally, TensorFlow Lite, being an inference engine, does not intrinsically support missing value imputation or categorical encoding during runtime. All such operations must be baked into the TFLite model itself or handled by pre-processing code *prior* to feeding data to the model. This constraint stems from the primary objective of minimizing the inference footprint on the microcontroller. The lack of runtime support compels a shift in how machine learning models are prepared for deployment in resource-limited environments compared to general-purpose computation. The most common methodology involves pre-processing during the model building and training phases and designing inference logic with data characteristics in mind.

Regarding missing values, the approach hinges entirely on the nature of the data and the chosen model's tolerance to incomplete observations. Instead of imputation within the TFLite model, missing values must be addressed during model training. Simple techniques such as mean or median imputation work effectively, particularly when missingness is not excessive. For example, if sensor data exhibits intermittent dropouts, replacing the missing value with the average value of the feature across the training dataset, or across a historical sliding window for time-series data, can yield reasonable performance without adding complexity to the inference pipeline.

Alternatively, and often preferable depending on the machine learning model architecture, an explicit ‘missing value’ indicator can be incorporated as a new feature. This avoids unintentionally introducing bias, such as the mean representing a non-existent value and potentially biasing predictions. Creating an indicator binary feature, which registers '1' if the value is missing, and '0' otherwise, effectively encodes missingness in the input. The model, during training, can then learn the relationship between missingness and the target outcome. This approach often yields more accurate predictions than simply using mean imputation when the data has specific patterns of missingness. Crucially, the same imputation or indicator logic, encoded within the TFLite interpreter's `input_tensor` assignment, must be reproduced during runtime on the microcontroller, before passing the input data to the model.

Concerning categorical variables, direct representation within the model input is incompatible with the typical numerical input format expected by TFLite. Here, label encoding, which maps categorical values to numerical integers, is generally inadequate for non-ordinal categories, as the numerical values introduce a false sense of ordering. One-hot encoding is superior, representing each category as a binary vector, where '1' denotes the presence of the specific category and '0' the absence. However, one-hot encoding can lead to substantial increases in the size of the model's input layer, a critical consideration given the limited memory capacity of microcontrollers.

To manage the input dimension increase resulting from one-hot encoding, several strategies become useful. One effective approach is to limit the one-hot encoded features to the most frequent categories while representing all other less frequent categories with a single 'other' encoding. This reduction can be achieved by analyzing the training data and identifying the most significant categories, followed by one-hot encoding of those, and encoding all less frequent occurrences using a single bit. Alternatively, if the categorical features exhibit a high degree of correlation with other features, dimension reduction techniques like feature selection or principal component analysis (PCA) during model training might be employed. Again, the data transformation needs to happen before feeding data to the TFLite interpreter.

Here are three illustrative code examples, assuming you are using Python with TensorFlow (for model creation) and C/C++ (for microcontroller deployment) for demonstration.

**Example 1: Mean Imputation and Model Creation (Python)**

```python
import tensorflow as tf
import numpy as np

# Assume training data with missing values
train_data = np.array([[1, 2, np.nan, 4],
                      [5, np.nan, 7, 8],
                      [9, 10, 11, np.nan],
                      [13, 14, 15, 16]], dtype=np.float32)
mean_values = np.nanmean(train_data, axis=0)  # Calculate mean across columns
imputed_train_data = np.nan_to_num(train_data, nan=mean_values)

# Sample labels
train_labels = np.array([0, 1, 0, 1], dtype=np.int32)

# Define a simple model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(imputed_train_data, train_labels, epochs=100)

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
  f.write(tflite_model)
```

*This example demonstrates mean imputation using `numpy.nanmean`. `numpy.nan_to_num` fills missing values using calculated column means. The model architecture is basic, but can be replaced with models more appropriate for prediction tasks, such as linear regression, logistic regression, or ensemble models. The key takeaway is that the missing values are handled *before* the model is trained and converted to TFLite format.*

**Example 2: Missing Value Indicator and Inference Logic (C++)**

```cpp
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include <vector>
#include <iostream>

// Placeholder for TFLite model loading
std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;

void loadModel(){
    // In a real scenario this would be loading from Flash.
    model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();
}

std::vector<float> predict(std::vector<float> rawInput) {
    // Impute using indicator method (as described)
    int numFeatures = rawInput.size();
    std::vector<float> processedInput;
    for(int i=0; i< numFeatures; i++){
      if(isnan(rawInput[i])){
          processedInput.push_back(0); // Raw Feature Value will be 0 as per imputation logic below
          processedInput.push_back(1); // missing indicator is 1 if value was missing.
       } else {
          processedInput.push_back(rawInput[i]);
          processedInput.push_back(0);
       }

    }

    // Assumes the model expects the processed input
    float* input_tensor = interpreter->typed_input_tensor<float>(0);
    std::copy(processedInput.begin(), processedInput.end(), input_tensor);

    interpreter->Invoke();

    float* output_tensor = interpreter->typed_output_tensor<float>(0);
    std::vector<float> output(output_tensor, output_tensor + interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float));
    return output;

}

int main() {
  loadModel();
  std::vector<float> input1 = {1, 2, NAN, 4}; // Sample input with a missing value
  std::vector<float> output1 = predict(input1);
  std::cout << "Output 1: ";
  for (float val : output1) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  std::vector<float> input2 = {5, 6, 7, 8};  // Sample input with no missing value
  std::vector<float> output2 = predict(input2);
  std::cout << "Output 2: ";
  for (float val : output2) {
    std::cout << val << " ";
  }
  std::cout << std::endl;


  return 0;
}
```

*This C++ code snippet demonstrates how to load a TFLite model and apply an indicator method for missing values before performing inference. It assumes the model expects twice the input features. Each input feature is replaced with two features: first one representing the actual raw data and next one representing the missing indicator (0 or 1). The specific missing value handling will depend on how the model was trained.*

**Example 3: One-Hot Encoding for Category Data and Inference Logic (Python)**
```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Sample categorical data
train_categories = np.array([['A'], ['B'], ['C'], ['A'], ['D'], ['B'] ])

encoder = OneHotEncoder(handle_unknown='ignore') # Ignore unseen categories at test time
encoded_train_categories = encoder.fit_transform(train_categories).toarray()
print (encoded_train_categories)
# Sample numerical data
train_numerical = np.array([[1], [2], [3], [4], [5], [6]], dtype=np.float32)

# Concatenate one hot encoded features and numerical features
train_data = np.concatenate((encoded_train_categories,train_numerical), axis=1)
# Sample labels
train_labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

# Define a simple model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, activation='relu', input_shape=(train_data.shape[1],)),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=100)

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("categorical_model.tflite", "wb") as f:
  f.write(tflite_model)

# Example inference

test_category = np.array([['B']]) # Sample testing category
encoded_test_category = encoder.transform(test_category).toarray()

test_numerical = np.array([[7]], dtype=np.float32) # Sample testing numerical data

test_input = np.concatenate((encoded_test_category,test_numerical), axis=1)
print(test_input)
```

*This python example demonstrates One-Hot Encoding for categorical variables using `sklearn.preprocessing.OneHotEncoder`. The one hot encoded values are concatenated to other numerical data, and this joint vector is used to train the model. Also, the example shows how to use the same encoder object to encode a test category before performing prediction using the model (this is a very important point in the context of microcontroller inference.)*

For further study, I would recommend reviewing documentation regarding:

1.  **TensorFlow Lite for Microcontrollers:** Provides a comprehensive understanding of the library and its constraints.
2.  **Feature Engineering techniques:** Specifically missing value imputation and categorical data encoding in the context of machine learning.
3.  **Model optimization strategies** for resource-constrained devices, such as model pruning and quantization.

Effective deployment of machine learning on microcontrollers hinges on careful preprocessing during model training, awareness of TFLite's limitations, and optimized runtime logic, which is designed to take into consideration specific data features and preprocessing choices. These considerations allow to generate useful inferences despite the constraints imposed by low-resource devices.
