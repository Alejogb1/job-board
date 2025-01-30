---
title: "How can digits be classified sequentially from left to right?"
date: "2025-01-30"
id: "how-can-digits-be-classified-sequentially-from-left"
---
Classification of digits sequentially from left to right, particularly when dealing with variable-length inputs, necessitates a stateful approach. Unlike classifying digits in isolation, understanding the contextual relationships formed by their position requires processing each digit in relation to its predecessor. This shifts the problem from simple pattern recognition to a sequence learning task.

My experience developing a machine-reading system for scanned invoices revealed that simple, independent digit classification resulted in numerous errors, especially with handwritten digits. The position of a ‘1’, for example, might be misclassified as a ‘7’ or a ‘4’ when considered in isolation. However, when considered as part of a string, “123” for example, and by tracking the transitions between each digit, the model’s accuracy improved significantly. This demonstrated that sequential processing, while adding complexity, provided critical context for accurate classification. I utilized recurrent neural networks, specifically Long Short-Term Memory (LSTM) networks, for this purpose and achieved a robust solution.

The crux of the problem lies in how to maintain and utilize the contextual information of prior digits. Treating the input not as a set of independent pixels, but as a series of values, is pivotal. The first step is to prepare the image data, extracting each digit and representing it numerically. This can be done using image segmentation techniques, followed by feature extraction via techniques such as histograms of oriented gradients (HOG), or directly feeding pixel data. Crucially, each extracted digit is not treated as the target for classification. Instead, it becomes a temporal input of the input sequence to be classified.

The following approach, although demonstrated with Python using Keras, could be adapted to different frameworks and languages:

**Code Example 1: Data Preparation and Sequence Creation**

```python
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def create_sequences(digits, labels, max_length):
    padded_sequences = []
    padded_labels = []
    for i, digit_set in enumerate(digits):
        if len(digit_set) <= max_length:
            padded_digit_set = np.pad(digit_set, (0, max_length - len(digit_set)), 'constant')
            padded_sequence = to_categorical(padded_digit_set, num_classes=10)
            padded_sequences.append(padded_sequence)
            padded_labels.append(labels[i])
    return np.array(padded_sequences), np.array(padded_labels)

# Assuming 'digits' is a list of lists, where each inner list contains the numerical representation of
# a sequence of digits (e.g., [[1,2,3], [4,5], [6,7,8,9]]).
# Assume 'labels' is a list of corresponding ground truth values for each sequence (e.g. [321, 54, 9876])
# For demonstration, generate some dummy data:

digits = [ [1,2,3], [4,5], [6,7,8,9], [0], [1,2], [9, 8, 7, 6, 5, 4, 3, 2, 1] ]
labels = [ 123, 45, 6789, 0, 12, 987654321 ]
max_length = max(len(seq) for seq in digits) # determines maximum sequence length
# Convert digits to categorical representation and create padded sequences
padded_sequences, padded_labels = create_sequences(digits, labels, max_length)

# Split data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, padded_labels, test_size=0.2, random_state=42)


print("Padded Sequence shape:", X_train.shape)
print("Padded Labels shape:", y_train.shape)
```

This first code example illustrates a critical step: data preparation. The `create_sequences` function takes a list of variable length digit sequences, pads the sequences to a uniform maximum length using zero-padding, and converts each digit to a one-hot encoded vector. This results in a 3D numpy array shaped as `(number_of_sequences, max_length, number_of_classes)` suitable for a recurrent neural network. For dummy data, the example generates sample digit sequences, applies padding to ensure uniform lengths, and converts numerical digits to one-hot encoded vector representations.  The data is then split into train and test sets. The printed shapes provide insight into how the data has been transformed for use with the model.

The next step involves defining the recurrent neural network architecture. The choice of LSTM layers over simple recurrent layers is due to their capacity to address vanishing gradient issues and capture long-term dependencies. The model takes the padded sequences as inputs, passes them through the LSTM layers and finally predicts the class at each time-step by applying a dense layer and a softmax activation.

**Code Example 2: Recurrent Neural Network Architecture**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed

def create_lstm_model(max_length, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_length, num_classes), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Creating model
num_classes = 10 # Number of output classes
model = create_lstm_model(max_length, num_classes)
print(model.summary())

```

This code defines an LSTM-based model tailored for sequential digit classification. The `create_lstm_model` function constructs a sequential model with two LSTM layers, and a TimeDistributed Dense layer. The `TimeDistributed` layer applies the dense layer to each time-step of the sequence.  Using a softmax activation allows the model to provide a probability distribution over all possible classes (0-9), for each time-step. The model is compiled using the Adam optimizer, categorical cross-entropy loss function (appropriate for multi-class classification), and accuracy as an evaluation metric. The summary provides an overview of the model structure, the output of each layer, and the number of trainable parameters.

The training step proceeds by feeding the padded sequences and corresponding labels to the model.  After training, model performance must be evaluated using unseen data from the test set.  This includes not just predicting each digit, but also evaluating how well the predictions align with the ground truth label, considering sequence alignment.

**Code Example 3: Training and Evaluation**

```python
# Reshape target labels for TimeDistributed layer
y_train_cat = np.zeros((len(y_train), max_length, num_classes))
y_test_cat = np.zeros((len(y_test), max_length, num_classes))

for i, seq in enumerate(y_train):
   temp = [int(digit) for digit in str(seq)]
   for j, d in enumerate(temp):
     y_train_cat[i, j, d] = 1

for i, seq in enumerate(y_test):
   temp = [int(digit) for digit in str(seq)]
   for j, d in enumerate(temp):
     y_test_cat[i, j, d] = 1

# Train the model
model.fit(X_train, y_train_cat, epochs=10, validation_split=0.2, verbose = 0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=-1)
print(f"Predicted labels shape: {predicted_labels.shape}")
```

This code demonstrates training the model and evaluating its performance. The target labels are converted to a format compatible with the TimeDistributed output of the model. Each digit in the sequence of ground truth is one-hot encoded. Then, the model is trained using `model.fit` for a fixed number of epochs, with a portion of the training data set aside for validation during training. The performance is then evaluated using `model.evaluate` on the test set. After training, the model provides prediction of the probability for each class at each time-step. By taking `argmax` over the probability, the predicted labels are extracted. The shape of the output illustrates that for each input sequence, the model produces a sequence of predictions with length equal to the padded sequence length.

For a more comprehensive understanding, I recommend exploring texts on recurrent neural networks, specifically those detailing sequence-to-sequence modeling. Additionally, resources covering time series analysis offer valuable insights into managing sequential data. Studying materials on image processing, especially those concerning image segmentation, is also highly beneficial in order to extract digits from images as an initial step. Finally, materials detailing data preprocessing for machine learning, including padding techniques for variable length sequences and one-hot encoding, can improve understanding of this subject.
