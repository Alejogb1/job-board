---
title: "How can I implement MSE as an evaluation metric for NNI models in PyTorch?"
date: "2025-01-30"
id: "how-can-i-implement-mse-as-an-evaluation"
---
My experience training neural networks has repeatedly demonstrated the critical importance of choosing the right evaluation metric. When using Neural Network Intelligence (NNI) for hyperparameter tuning, ensuring your chosen metric aligns with your problem is paramount. You're asking specifically about Mean Squared Error (MSE), a common loss function, but crucially, also valuable as an evaluation metric.  The challenge lies in integrating it effectively within the NNI workflow, distinct from its usage as a training objective. Here's a breakdown of how to achieve this, building upon the assumption you're using PyTorch for your model implementation.

The core difference to understand is that the training process uses MSE, or another loss, to *optimize* the model parameters. Evaluation, on the other hand, gauges how well the model performs on data it hasn’t seen during training. While we might *train* on MSE, we might still want to *evaluate* performance using metrics like accuracy in a classification problem or, as in your case, MSE for regression. NNI needs a metric to guide its hyperparameter search, and it's often more insightful to use an evaluation metric based on separate validation data. Therefore, it's important to compute the metric independently, separate from training step, usually after completing a training epoch, and then report this score to NNI.

Here's how it’s implemented within an NNI environment for a PyTorch model:

First, let's define the components of a typical NNI trial function. The crucial part is calculating the MSE on validation data *after* training and then reporting it back to NNI.

```python
import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train(model, train_dataloader, optimizer, criterion, device):
    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def validate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


def create_dataset(size=1000, dim=10, batch_size=32):
    inputs = torch.randn(size, dim)
    targets = torch.randn(size, 1)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_model(input_dim=10, hidden_dim=50):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )


def trial_function(params):
    # 1. Load the NNI configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = params['learning_rate']
    hidden_dim = params['hidden_dim']

    # 2. Define the model, optimizer, loss function.
    model = create_model(hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Data loading.
    train_dataloader = create_dataset()
    val_dataloader = create_dataset(size=500)


    # 4. Train the model.
    for epoch in range(2):  # Keeping it short for demonstration
        train(model, train_dataloader, optimizer, criterion, device)

    # 5. Calculate the validation MSE
    validation_mse = validate(model, val_dataloader, criterion, device)


    # 6. Report the MSE to NNI. This is the key step!
    nni.report_final_result(validation_mse)



if __name__ == '__main__':
    search_space = {
        'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'hidden_dim': {'_type': 'randint', '_value': [20, 100]},
    }

    nni.run(
        trial_function,
        search_space,
        max_trial_number=10,
        optimize_mode='minimize'
    )

```

**Code Explanation:**

*   The `trial_function` is the heart of our NNI experiment, receiving hyperparameter suggestions (`params`).
*   We construct our model (`create_model`) and dataloaders (`create_dataset`). Notice the creation of both train and validation dataloaders for proper evaluation.
*   The training loop is fairly standard, optimizing the model using a learning rate provided by the `params` dictionary.
*   Crucially, after training, we call `validate()` which computes MSE on the validation dataset. The validation MSE is then extracted as a single, scalar value using the `item()` method, before reporting.
*   Finally, `nni.report_final_result()` sends this calculated MSE back to NNI. NNI uses this value to decide what set of hyperparameters to try next.
*   We define a basic search space and invoke the NNI framework through the `nni.run` command, setting `optimize_mode` to `minimize` as our goal is to minimize MSE.

A common source of errors is directly reporting the entire loss value from the last training iteration. This is often not representative of the overall validation performance. It's essential to process the entire validation set to obtain a stable representation of the model's generalization.

Let's elaborate on a slightly different situation, where you might calculate the MSE manually instead of relying on `nn.MSELoss`, perhaps because you need additional processing of the output:

```python
import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train(model, train_dataloader, optimizer, criterion, device):
    # (Same as before)
    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def validate_manual_mse(model, val_dataloader, device):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = torch.mean((all_outputs - all_targets)**2)
    return mse.item()

def create_dataset(size=1000, dim=10, batch_size=32):
    # (Same as before)
    inputs = torch.randn(size, dim)
    targets = torch.randn(size, 1)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def create_model(input_dim=10, hidden_dim=50):
    # (Same as before)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

def trial_function(params):
    # 1. Load the NNI configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = params['learning_rate']
    hidden_dim = params['hidden_dim']

    # 2. Define the model, optimizer, loss function.
    model = create_model(hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()  # Still use MSE for training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Data loading.
    train_dataloader = create_dataset()
    val_dataloader = create_dataset(size=500)


    # 4. Train the model.
    for epoch in range(2):  # Keeping it short for demonstration
        train(model, train_dataloader, optimizer, criterion, device)

    # 5. Calculate the validation MSE manually.
    validation_mse = validate_manual_mse(model, val_dataloader, device)


    # 6. Report the MSE to NNI. This is the key step!
    nni.report_final_result(validation_mse)


if __name__ == '__main__':
    search_space = {
        'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'hidden_dim': {'_type': 'randint', '_value': [20, 100]},
    }

    nni.run(
        trial_function,
        search_space,
        max_trial_number=10,
        optimize_mode='minimize'
    )
```

**Code Explanation:**

* The `validate_manual_mse` function now collects all outputs and targets and calculates the MSE directly.
* Notice that this does not change how `nn.MSELoss` is used in training.
* This offers increased flexibility if post-processing or custom MSE logic is required for the evaluation.

Finally, if you're dealing with a situation where you want to report multiple metrics to NNI simultaneously, it can be accomplished as follows, using the `nni.report_intermediate_result` call instead:

```python
import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
def train(model, train_dataloader, optimizer, criterion, device):
  # (Same as before)
    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def validate_manual_mse(model, val_dataloader, device):
  # (Same as before)
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mse = torch.mean((all_outputs - all_targets)**2)
    return mse.item(), torch.mean(torch.abs(all_outputs - all_targets)).item()


def create_dataset(size=1000, dim=10, batch_size=32):
    # (Same as before)
    inputs = torch.randn(size, dim)
    targets = torch.randn(size, 1)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_model(input_dim=10, hidden_dim=50):
  # (Same as before)
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )



def trial_function(params):
    # 1. Load the NNI configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = params['learning_rate']
    hidden_dim = params['hidden_dim']

    # 2. Define the model, optimizer, loss function.
    model = create_model(hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Data loading.
    train_dataloader = create_dataset()
    val_dataloader = create_dataset(size=500)


    # 4. Train the model.
    for epoch in range(2):  # Keeping it short for demonstration
        train(model, train_dataloader, optimizer, criterion, device)

        # 5. Calculate the validation metrics after each epoch.
        validation_mse, validation_mae = validate_manual_mse(model, val_dataloader, device)
        nni.report_intermediate_result({
            'mse': validation_mse,
            'mae': validation_mae,
         })

    # 6. Report the *final* MSE to NNI.
    validation_mse, _  = validate_manual_mse(model, val_dataloader, device)
    nni.report_final_result(validation_mse)



if __name__ == '__main__':
    search_space = {
        'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'hidden_dim': {'_type': 'randint', '_value': [20, 100]},
    }

    nni.run(
        trial_function,
        search_space,
        max_trial_number=10,
        optimize_mode='minimize'
    )
```

**Code Explanation:**

* The `validate_manual_mse` function now returns two values: MSE and Mean Absolute Error.
* After each training epoch we report both the intermediate results of MSE and MAE to NNI.
* NNI uses the value reported with `report_final_result` for its optimization, as specified by the `optimize_mode`, but it does keep track of the intermediate metrics as well.
* If you have the necessity to track performance after each epoch, but you would still like NNI to base its hyperparameter search on a *final* evaluation metric, this structure will suit your needs.

For resources, I recommend exploring the official PyTorch documentation for MSE loss and data loading, as well as the NNI documentation. Further, looking at examples from the NNI GitHub repository can offer valuable insights. Specifically focusing on the `nni.report_final_result` function within the API reference will clarify exactly what is needed for the optimization loop. Consulting academic resources on regression and evaluation metrics would also prove helpful, specifically delving into the differences between training loss and evaluation metrics. Understanding that loss during training is for the optimization process, and the evaluation metrics are for gauging generalization, will result in a more effective workflow with NNI. Remember to carefully select the most appropriate evaluation metric for your specific problem.
