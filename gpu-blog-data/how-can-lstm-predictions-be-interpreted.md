---
title: "How can LSTM predictions be interpreted?"
date: "2025-01-30"
id: "how-can-lstm-predictions-be-interpreted"
---
Long Short-Term Memory (LSTM) networks, while powerful for sequential data modeling, often present a “black box” challenge when it comes to interpreting their predictions. The network’s internal state and complex interactions make it difficult to directly ascertain which input features or time steps contributed most significantly to a particular output. However, understanding these contributions is crucial for building trust in the model, debugging unexpected behavior, and potentially gaining new insights into the underlying data patterns. My experience working with LSTMs in financial time series analysis has repeatedly underscored this challenge.

The difficulty in interpreting LSTM predictions stems primarily from two factors: the recurrent nature of the architecture and the non-linear activation functions employed. Recurrent networks, by design, maintain a hidden state that summarizes the sequence information encountered up to a particular point. This hidden state is then fed back into the network at the next time step, making the impact of each input point difficult to isolate. Non-linear functions like sigmoid or tanh further complicate the matter, as they transform the linear combinations of input features into non-linear outputs, obscuring the direct contribution of individual input features.

Therefore, effective interpretation requires shifting from a direct input-output perspective to one focused on identifying influencing factors and attributing importance to various parts of the input sequence. A range of methods have been developed to approach this, which can broadly be categorized into sensitivity analysis, attention mechanisms, and feature importance techniques.

Sensitivity analysis focuses on perturbing input features and observing the resulting change in output. By systematically changing input values or masking specific time steps, we can assess how sensitive the network's predictions are to those changes. In practical terms, this might mean evaluating the effect of removing a particular feature across the sequence or introducing noise into its value and monitoring the impact on the model’s forecast. If a small perturbation leads to a significant change, that input is likely to play an important role in generating the final prediction. However, it is computationally intensive to conduct a thorough analysis of all possible feature combinations.

Attention mechanisms provide an alternative strategy by allowing the network to learn which portions of the input sequence are most relevant for generating a particular prediction. These mechanisms assign weights to each time step in the input sequence, effectively emphasizing the more significant parts while downplaying the less critical ones. The learned attention weights are directly interpretable and offer a visualization of what the network considers most relevant during the decision-making process. Critically, the attention weights are also learned in conjunction with the task, avoiding the potentially cumbersome evaluation process required by sensitivity analysis. It is worth mentioning that there exist several forms of attention, from the original attention used in neural machine translation to more complex variants like multi-headed attention.

Feature importance techniques, while less direct in the temporal sense, aim to determine which input features exert the strongest influence across the whole sequence. Methods like SHAP (SHapley Additive exPlanations) values provide a framework for attributing the output of a machine learning model to the features that contribute to it. The Shapley value computation is computationally expensive, but the output of these calculations is a set of feature importance scores. These scores represent the average impact of each feature across all possible input combinations. The limitation in the temporal context is that these approaches often consider feature contributions at an aggregate level rather than specific to individual time steps.

To illustrate these interpretation approaches with code examples, I will use a hypothetical LSTM model trained to predict stock price fluctuations. Assume an input sequence of five historical days, with three input features: open price, close price, and trading volume.

**Example 1: Sensitivity Analysis**

This example implements a simple sensitivity analysis focusing on the impact of perturbing a single feature.

```python
import numpy as np
import tensorflow as tf

# Assume model is a pre-trained LSTM model
# and input_data is a numpy array of shape (1, 5, 3)


def sensitivity_analysis(model, input_data, feature_index, perturbation_size=0.1):
    """Performs sensitivity analysis by perturbing a specific feature."""
    original_prediction = model.predict(input_data)[0]
    perturbed_input = np.copy(input_data)
    perturbed_input[:, :, feature_index] = perturbed_input[:, :, feature_index] * (1 + perturbation_size)
    perturbed_prediction = model.predict(perturbed_input)[0]
    sensitivity = np.abs(perturbed_prediction - original_prediction)

    return sensitivity


# Example Usage
#  model is a tf.keras.Model
# input_data shape is (1, 5, 3)
# Feature indexes: 0=open price, 1=close price, 2=volume
input_data = np.random.rand(1, 5, 3)
model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=32, input_shape=(5,3), return_sequences=False),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
model.compile(optimizer='adam', loss='mse')
open_price_sensitivity = sensitivity_analysis(model, input_data, 0)
close_price_sensitivity = sensitivity_analysis(model, input_data, 1)
volume_sensitivity = sensitivity_analysis(model, input_data, 2)
print("Open Price Sensitivity:", open_price_sensitivity)
print("Close Price Sensitivity:", close_price_sensitivity)
print("Volume Sensitivity:", volume_sensitivity)
```

This function takes an LSTM model, input data, and the feature index as input, then perturbs the values of the specified feature across the input sequence. The output sensitivity is the absolute difference between the perturbed and the original prediction. This function is run multiple times to test sensitivity to each feature.

**Example 2: Attention Mechanism (Conceptual)**

This example illustrates the conceptual structure of incorporating an attention mechanism. In this simplified case, the attention is applied after the LSTM layer output.

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class AttentionLayer(Layer):
    """Conceptual Attention Layer."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
                                initializer="random_normal", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        attention_weights = tf.nn.softmax(tf.matmul(x, self.W), axis=1) #shape is (None, 5, 1)
        weighted_x = x * attention_weights  # shape is (None, 5, input_size)
        return tf.reduce_sum(weighted_x, axis=1)

# Create an LSTM model with attention
model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=32, input_shape=(5, 3), return_sequences=True),
        AttentionLayer(),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])

model.compile(optimizer='adam', loss='mse')

# Example usage:
input_data = np.random.rand(1, 5, 3)
predictions = model.predict(input_data)
print("Model prediction", predictions)
attention_weights = model.layers[1].W
print("Attention weights:", attention_weights)

```
This demonstrates the fundamental concept. After the LSTM layer, an attention layer calculates weights for each time step. These weights are used to generate a weighted sum of the LSTM output, then fed into the final dense layer. The `attention_weights` attribute is directly interpretable. In a more complex real-world implementation, multiple attention heads would be used.

**Example 3: Feature Importance using SHAP (Conceptual)**

This example shows the idea behind using SHAP values to understand the influence of each feature. This relies on an external library.

```python
import shap
import numpy as np
import tensorflow as tf

# Example usage with shap
#  model is a tf.keras.Model with output shape (1,)

# Define the baseline for shap. We assume that input shape is (1, 5, 3)
background = np.zeros((100, 5, 3))
model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=32, input_shape=(5,3), return_sequences=False),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
model.compile(optimizer='adam', loss='mse')

# Wrap the model to output numpy arrays
def wrapped_predict(input_data):
    return model.predict(input_data).flatten()

explainer = shap.KernelExplainer(wrapped_predict, background) # note the model needs to output a numpy array to work with shap
input_data = np.random.rand(1, 5, 3)
shap_values = explainer.shap_values(input_data) # The output will be (1, 15)
feature_names = ["Open_t-4", "Close_t-4", "Volume_t-4",
                 "Open_t-3", "Close_t-3", "Volume_t-3",
                 "Open_t-2", "Close_t-2", "Volume_t-2",
                 "Open_t-1", "Close_t-1", "Volume_t-1",
                 "Open_t", "Close_t", "Volume_t"]

print("SHAP values shape:", shap_values.shape)
shap.summary_plot(shap_values, feature_names=feature_names)


```
This example utilizes the SHAP library to attribute the output to different input features. The background data is created, then the model is wrapped to return numpy arrays. The shap values are computed. In this case, the importance of each feature at each time step would be shown, if printed to the terminal.

For further exploration, I would recommend exploring the official documentation for the Keras and Tensorflow libraries. The book “Interpretable Machine Learning” by Christoph Molnar provides a comprehensive overview of various interpretation techniques, including those discussed here. Also, research papers focusing on attention mechanisms in deep learning can offer deeper insight. Publications from venues such as NeurIPS and ICLR are excellent sources for cutting-edge methods. Specifically, look for papers on time-series analysis with recurrent neural networks.
