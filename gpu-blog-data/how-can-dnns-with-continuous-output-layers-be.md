---
title: "How can DNNs with continuous output layers be fine-tuned?"
date: "2025-01-30"
id: "how-can-dnns-with-continuous-output-layers-be"
---
Fine-tuning Deep Neural Networks (DNNs) with continuous output layers, specifically those predicting numerical values rather than discrete classes, requires a nuanced approach compared to classification tasks. The primary distinction lies in the loss functions employed and the metrics used to evaluate performance. My experience building predictive models for high-frequency financial time series highlighted these critical differences, where even small miscalibrations could lead to substantial financial losses.

The challenge with continuous output layers is that we're not optimizing for correct classification, but for minimizing the discrepancy between predicted values and true values. This fundamentally shifts the goal of fine-tuning from a discrete matching problem to a continuous approximation problem. Typically, a pretrained model, having learned robust features on a large dataset, is adapted to a new task with a smaller, task-specific dataset. In this scenario, a model outputting numerical values is fine-tuned to a specific regression task.

**1. Fine-tuning Process and Key Considerations**

The fine-tuning procedure involves a number of steps. First, a pre-trained model, often trained on a very large image or text dataset, is obtained. The final classification layer is then removed and replaced with a new layer suitable for regression. This could be a single dense layer with a linear activation function to output a single numerical value, or multiple layers to output a vector of numbers. The architecture of the new regression layers is often determined based on the complexity of the target function. After the model is adapted, the final model can be trained on the new dataset. 

The next crucial aspect is the choice of the loss function. Unlike classification tasks, where cross-entropy is prevalent, regression tasks typically utilize measures like Mean Squared Error (MSE), Mean Absolute Error (MAE), or Huber Loss. MSE, for instance, is sensitive to outliers, potentially leading the model to over-emphasize extreme values during training. MAE is more robust to outliers but may lack the smoothness needed for optimization. Huber Loss seeks a compromise, acting as MSE for small errors and as MAE for large errors, and thus may be an effective choice. The choice of loss function can be crucial for the resulting model performance, with each having tradeoffs in terms of sensitivity to outliers, gradient stability, and convergence.

Furthermore, the learning rate must be carefully controlled. Using the same learning rate used during the initial pre-training phase can be detrimental, as the model is often already close to an acceptable solution. A smaller learning rate is typical for fine-tuning, allowing for more gradual adjustments to the parameters. Adaptive learning rate algorithms like Adam or RMSprop can be helpful by automatically adjusting the learning rate per parameter, and these may prove useful when the model is adapting from a broad pre-training dataset to a more narrow, task-specific domain. The learning rate should also be smaller for the initial layers. Freezing the early layers of the network and fine-tuning only the newly added regression layers is another common practice. This technique avoids disrupting the robust feature representations already learned in the earlier layers and provides better training when small amounts of new data are available. 

Finally, appropriate evaluation metrics must be selected. While loss functions guide the training process, metrics provide a human-interpretable way of evaluating the model's performance. For regression tasks, metrics like Root Mean Squared Error (RMSE), R-squared (coefficient of determination), or mean absolute percentage error (MAPE) are common options. RMSE is a metric of model performance, while R-squared shows how much variance is explained by the model. MAPE can be useful to show percentage error. The choice of metrics will depend on the context of the task. For example, a model predicting house prices should be assessed using an absolute value metric like RMSE, whereas the accuracy of a model predicting the volatility of an asset may be measured with a metric like MAPE.

**2. Code Examples**

Below are three code examples, written with Python and TensorFlow's Keras API. They illustrate different aspects of fine-tuning:

**Example 1: Fine-tuning with a linear output and Mean Squared Error**

```python
import tensorflow as tf
from tensorflow.keras import layers

# Load a pre-trained model (e.g., ResNet50) - assume weights are available
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
  layer.trainable = False

# Add a custom regression head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) # Additional layer with nonlinear activation
predictions = layers.Dense(1, activation='linear')(x) # Linear output for regression

# Create the fine-tuned model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='mean_squared_error', 
              metrics=['mean_absolute_error'])

# Assume train_data and train_labels are defined
# model.fit(train_data, train_labels, epochs=10, batch_size=32) # train the model
```
**Commentary:** This example uses ResNet50 as a pre-trained model. We freeze the pre-trained layers and add a Global Average Pooling layer followed by a fully connected layer with ReLU activation. Finally, we add a linear output layer for regression. The `mean_squared_error` loss function and `mean_absolute_error` metric are used. The training rate is kept small during fine-tuning with the `Adam` optimizer to not disrupt the base model's performance.

**Example 2: Fine-tuning with Mean Absolute Error (MAE) Loss**

```python
import tensorflow as tf
from tensorflow.keras import layers

# Assume a pre-trained model 'pretrained_model' is already loaded
pretrained_model = tf.keras.models.Sequential([
  layers.Dense(64, activation='relu', input_shape=(100,)), 
  layers.Dense(32, activation='relu'),
])

# Add a regression head (outputting a vector of 3 numerical values)
x = pretrained_model.output
predictions = layers.Dense(3, activation='linear')(x)
model = tf.keras.Model(inputs=pretrained_model.input, outputs=predictions)

# Compile the model using MAE
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

# Assuming 'x_train' and 'y_train' datasets are prepared
# model.fit(x_train, y_train, epochs=10, batch_size=32) # train the model
```
**Commentary:** This example illustrates using MAE. The pre-trained model has two layers of dense networks. The final output layer has three nodes with linear activations, suitable for multiple regression. The loss and the metric are set to MAE, which is robust to outliers. An adaptive optimizer is used with the default learning rate.

**Example 3: Fine-tuning using Huber loss and monitoring R-squared**

```python
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

# Assume a pre-trained Transformer model
input_tensor = tf.keras.Input(shape=(50,))
x = layers.Embedding(input_dim=1000, output_dim=64)(input_tensor)
x = layers.Transformer(num_heads=2, key_dim=32)(x, x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
pretrained_transformer = tf.keras.Model(inputs=input_tensor, outputs=x)

# Add regression output
x = pretrained_transformer.output
output = layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=pretrained_transformer.input, outputs=output)

# Define R-squared metric
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Compile the model with Huber loss
model.compile(optimizer='adam', loss='huber', metrics=[r_square])

# Assumed data
# model.fit(x_data, y_data, epochs=10, batch_size=32)
```
**Commentary:** This example demonstrates the use of a pre-trained transformer model. The Huber loss function is used for better stability against outliers. Also, a user-defined R-squared metric is used for performance evaluation, illustrating a complex regression problem where R-squared might be more informative than simpler metrics.

**3. Resource Recommendations**

To further improve fine-tuning practices, several resources should be consulted. For general deep learning techniques, including fine-tuning strategies, various textbooks offer detailed explanations of backpropagation, optimization, and regularization. These can improve understanding of core techniques. Also, many online courses provide practical examples and code walkthroughs for different deep learning tasks, including fine-tuning. This may help expand your understanding of specific modeling tasks. Finally, academic papers describing recent developments in fine-tuning methods can provide the most cutting-edge techniques. Reviewing both applied and theoretical work will provide a deeper understanding of the fine-tuning process.
