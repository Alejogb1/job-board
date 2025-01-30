---
title: "Why are RNN/GRU validation losses increasing while mean absolute errors decrease?"
date: "2025-01-30"
id: "why-are-rnngru-validation-losses-increasing-while-mean"
---
A discrepancy where validation loss increases while the mean absolute error (MAE) decreases during recurrent neural network (RNN) or Gated Recurrent Unit (GRU) training often indicates a misalignment between the loss function and the evaluation metric, a scenario I've encountered frequently in time-series forecasting projects. Specifically, this often signals the model is optimizing for a distribution of predictions different from the one that minimizes the average absolute deviation. The loss function guides the gradient descent, while the MAE evaluates the practical accuracy of the model's output.

The key problem lies in the nature of the loss function. Commonly, during training, Mean Squared Error (MSE) is employed as the loss function due to its differentiability properties and ease of use. MSE penalizes larger errors more severely, which can push the model toward predicting values that are statistically 'safe' in the sense of minimizing squared deviations, rather than focusing on minimizing the absolute differences. The model may learn to predict a slightly skewed distribution with a bias, thus achieving lower MSE but not necessarily lower MAE. Meanwhile, MAE treats all errors linearly. A slight shift in the predicted values, even if it minimizes the total sum of squared errors, might actually increase the total sum of absolute errors.

Consider a time-series prediction task where the true values are [10, 20, 30, 40]. An initial model might predict [12, 22, 28, 38]. Assume MSE calculates to be 20 and MAE to be 2. Subsequent training, influenced by MSE, might push the predictions to [11, 21, 29, 39]. Here, the MSE could decrease to 10 (hypothetically), while the MAE might remain at or even rise above 2. The reduction in squared errors doesn’t necessarily correlate to the same pattern of reduction in absolute errors. The model is simply finding a local minimum for MSE and not MAE. It’s not a problem with overfitting but a problem with the training signal; the gradients are not pointing in the optimal direction to minimize MAE.

This effect is further exacerbated in time series tasks because the time dependencies of the input data create more complex relationships for the model to learn than a standard regression problem with independent samples. RNNs and GRUs, with their internal states, can develop dependencies across time steps that lead to these nuanced shifts in predictions. The MSE calculation over a sequence of output from these models might improve overall, by penalizing larger errors, while individual absolute deviations may increase. The model may be smoothing out predictions, resulting in smaller squared differences, but also a less accurate average prediction. The ‘bias’ of the prediction becomes a more important factor in MSE calculation than an average error, while MAE directly penalizes the bias as a simple linear penalty.

Below are three code examples illustrating this concept, using a simplified framework with PyTorch to demonstrate:

**Example 1: Simplified Training Loop with Diverging Loss and MAE:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data
true_values = torch.tensor([10.0, 20.0, 30.0, 40.0]).reshape(1, 1, -1) #Batch, sequence, features
predictions = torch.tensor([12.0, 22.0, 28.0, 38.0]).reshape(1, 1, -1)

# Model - Simple linear transformation
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(4, 4) #input and output are same dimensions

    def forward(self, x):
        return self.linear(x.squeeze()).unsqueeze(1)

model = LinearModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()

for _ in range(100):
    optimizer.zero_grad()
    pred_output = model(predictions)
    mse_loss = mse_loss_fn(pred_output, true_values) #loss function
    mae_loss = mae_loss_fn(pred_output, true_values) #evaluation metric

    mse_loss.backward()
    optimizer.step()

    print(f"MSE Loss: {mse_loss.item():.4f}, MAE: {mae_loss.item():.4f}")

    predictions = pred_output.detach() #update prediction.
```
*Commentary:* This example sets up a simple linear transformation, approximating a basic model. I use MSE as the training loss and MAE for evaluation. By updating predictions in each iteration, we can simulate the progression during training. Observe that with MSE as the training driver, MAE may not decrease in tandem with MSE, despite gradients being based on MSE.

**Example 2: Demonstrating a Skewed Distribution with RNN:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Dummy data
true_values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]).reshape(1, 1, -1)
input_seq = torch.rand(1, 1, 7)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out.squeeze(1))
        return out.unsqueeze(1)

model = SimpleRNN(7, 16, 7)
optimizer = optim.Adam(model.parameters(), lr=0.01)
mse_loss_fn = nn.MSELoss()
mae_loss_fn = nn.L1Loss()

for _ in range(1000):
    optimizer.zero_grad()
    output = model(input_seq)
    mse_loss = mse_loss_fn(output, true_values)
    mae_loss = mae_loss_fn(output, true_values)

    mse_loss.backward()
    optimizer.step()

    print(f"MSE Loss: {mse_loss.item():.4f}, MAE: {mae_loss.item():.4f}")
```

*Commentary:* This example uses a simple RNN. Here, even when trained on MSE the model may not always reduce absolute error, instead learning a skewed prediction. The goal is to demonstrate that the divergence between MSE and MAE is not an unusual problem, especially with the dynamic nature of an RNN. The loss function guides toward an optimum that might not produce the best absolute deviations.

**Example 3: Changing Loss Function to MAE**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Dummy data
true_values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]).reshape(1, 1, -1)
input_seq = torch.rand(1, 1, 7)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out.squeeze(1))
        return out.unsqueeze(1)

model = SimpleRNN(7, 16, 7)
optimizer = optim.Adam(model.parameters(), lr=0.01)
mae_loss_fn = nn.L1Loss()

for _ in range(1000):
    optimizer.zero_grad()
    output = model(input_seq)
    mae_loss = mae_loss_fn(output, true_values)

    mae_loss.backward()
    optimizer.step()

    print(f"MAE: {mae_loss.item():.4f}")
```
*Commentary:* In this example, I switched the training loss to MAE directly. When using MAE as the loss, the gradients directly push the predictions to minimize absolute differences. This generally results in improvements to MAE as opposed to the prior examples with MSE. This clearly illustrates the cause of the problem, and the potential remedy.

To mitigate this issue, several approaches can be considered. First, using MAE as the training loss function directly is the most straightforward. However, MAE is not everywhere differentiable, which can cause problems with certain optimization techniques. Smoothed L1 loss (Huber loss), a hybrid between MSE and MAE, can be a good compromise. Additionally, one could use a custom loss function that incorporates both MSE and MAE, which allows the model to benefit from the properties of both the functions. Finally, techniques such as early stopping using the validation MAE as the criterion rather than validation loss can also be effective.

For resources, I suggest referring to advanced deep learning textbooks that cover loss function selection for time series data, especially in the context of RNN and GRU networks. Also, numerous research papers detail the implications of different loss functions for deep learning models. Articles exploring metric design for model performance evaluation would also be beneficial to review, particularly those that compare and contrast loss function and evaluation metrics for regression. Furthermore, open-source machine learning libraries like PyTorch or TensorFlow have numerous tutorials and example notebooks for training time-series models that often include different loss function strategies that can be investigated to further understanding of the concepts discussed. These resources will provide a strong foundation for understanding the nuances of loss function selection in deep learning and its impact on model performance.
