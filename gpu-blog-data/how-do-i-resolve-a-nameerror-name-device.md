---
title: "How do I resolve a NameError: name 'device' is not defined in PyTorch?"
date: "2025-01-30"
id: "how-do-i-resolve-a-nameerror-name-device"
---
The `NameError: name 'device' is not defined` in PyTorch typically stems from an attempt to utilize a `device` variable without prior assignment.  This isn't a PyTorch-specific error; rather, it's a fundamental Python issue indicating the interpreter hasn't encountered a definition for the identifier `device` within the current scope.  My experience troubleshooting this across numerous deep learning projects, ranging from image classification to natural language processing, points to a consistent set of causes and solutions.

**1.  Clear Explanation:**

The error arises because your code references `device` (e.g., `model.to(device)`) before the variable has been assigned a value representing the desired computational device (CPU or GPU).  Python's scoping rules dictate that variables must be defined before being used.  The interpreter, encountering `device` without a prior definition, raises the `NameError`. This often occurs when code intended to run on a GPU is executed without proper device selection.  Furthermore, improper module imports, particularly concerning PyTorch's `torch` module, can contribute to the problem.  Finally, subtle scoping issues within functions or classes can mask the source of the undefined variable, making debugging more challenging.


**2. Code Examples with Commentary:**

**Example 1: Incorrect Device Specification**

```python
import torch

# Incorrect: device is used before assignment
model = torch.nn.Linear(10, 2)
model.to(device)  # NameError: name 'device' is not defined

# Correct: Specify the device explicitly.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)
```

This example highlights the most common cause. The code attempts to move the model to the `device` before defining `device`. The corrected version explicitly checks for CUDA availability using `torch.cuda.is_available()` and assigns `device` accordingly, ensuring the model is placed on the appropriate device (GPU if available, otherwise CPU).


**Example 2: Scoping Issues within a Function**

```python
import torch

def train_model(model, data):
    # Incorrect: device is not defined in the function scope.
    model.to(device)
    # ... training loop ...

# Correct: Pass device as an argument or define it within the function
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = torch.nn.Linear(10,2)
train_model(model, some_data, device) #Added device as an argument.

#OR

def train_model(model, data):
  if torch.cuda.is_available():
    device = torch.device('cuda')
  else:
    device = torch.device('cpu')
  model.to(device)
  # ... training loop ...

model = torch.nn.Linear(10,2)
train_model(model, some_data)
```

This example demonstrates a scoping problem.  The `device` variable isn't defined within the `train_model` function's scope.  The correction involves either passing `device` as an argument to the function or defining it inside the function's body.  The latter approach encapsulates the device selection logic within the function, enhancing modularity.


**Example 3:  Incorrect Module Import (Less Common but Possible)**

```python
# Incorrect: Assume 'device' is part of another module and imported incorrectly
from some_other_module import * #Hypothetically importing device from another module
model = torch.nn.Linear(10,2)
model.to(device)

#Correct: Ensure direct import of torch or explicit declaration of device.
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = torch.nn.Linear(10,2)
model.to(device)
```

While less frequent, mistakenly relying on a wildcard import (`from some_other_module import *`) might lead to an undefined `device` if the intended module doesn't actually contain it.  The solution is to explicitly import the necessary modules (specifically `torch` here) and define `device` as before.  Avoid wildcard imports generally, as they can obscure dependencies and lead to naming conflicts.


**3. Resource Recommendations:**

The official PyTorch documentation provides comprehensive details on tensors, devices, and CUDA management. Consult the sections covering tensor manipulation and CUDA usage for a thorough understanding of device selection and management.  Furthermore, explore introductory materials on Python's scoping rules and variable assignments to solidify your understanding of variable lifecycles and visibility within different code blocks.  A thorough understanding of Python's object-oriented features will also assist in debugging complex scenarios, such as class-based model definitions.  Finally, explore resources on debugging techniques within a Python environment, including the use of print statements, debuggers, and IDE-integrated debugging tools to identify the root cause of similar issues in your future projects.  Remember consistent and systematic debugging practices are crucial for resolving such errors effectively.
