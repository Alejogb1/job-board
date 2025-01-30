---
title: "Why is 'defaults' undefined in my PyTorch code?"
date: "2025-01-30"
id: "why-is-defaults-undefined-in-my-pytorch-code"
---
The `NameError: name 'defaults' is not defined` in PyTorch typically arises from attempting to access a variable or attribute named `defaults` that hasn't been previously declared or imported into the current scope.  This is a common error stemming from a misunderstanding of Python's scoping rules and how PyTorch modules are structured.  My experience troubleshooting this across numerous projects, particularly those involving custom datasets and model architectures, points to three primary causes.

1. **Incorrect Module Import or Path:** The most frequent cause is an incorrect or missing import statement.  PyTorch doesn't inherently possess a global variable or attribute called `defaults`. If your code expects a `defaults` variable or dictionary containing hyperparameters, model configurations, or dataset settings, this variable must be explicitly defined or imported from a specific module where it's defined.  This frequently occurs when working with configurations loaded from files (e.g., JSON, YAML) or when attempting to access settings from a module you've created. For instance, if you have a configuration file named `config.py` with a `defaults` dictionary, you must explicitly import it:

   ```python
   # config.py
   defaults = {
       'learning_rate': 0.001,
       'batch_size': 32,
       'epochs': 100
   }

   # your_script.py
   from config import defaults

   model = MyModel(defaults['learning_rate'])
   # ... rest of your code ...
   ```

   Failure to include `from config import defaults` will result in the `NameError`.  The same principle applies if `defaults` resides within a class or function within another module.  Ensure that the module containing the `defaults` object is correctly imported and that the path to this module is accurately reflected in your import statement.


2. **Typographical Errors:** While seemingly trivial, simple typos are a significant source of this error.  Python is case-sensitive, meaning `defaults`, `Defaults`, and `DefaultS` are considered distinct variables.  A minor misspelling in the variable name during its declaration, usage, or import can lead to this error.  Furthermore, double-checking for incorrect spacing or extra characters around the variable name in the import statement is crucial.  Thorough code review, including careful examination of every instance where `defaults` is used, can prevent this.

3. **Incorrect Scope and Namespace:** This error can stem from misunderstanding Python's scope rules.  If `defaults` is defined within a function or class, it's only accessible within that specific scope.  Attempting to access it from outside that function or class will trigger the `NameError`. Similarly, if `defaults` is defined in a nested function or closure, the outer functions might not have access to it.

   Consider this example, where `defaults` is only accessible within the `train_model` function:

   ```python
   def train_model(learning_rate):
       defaults = {'learning_rate': learning_rate}
       model = MyModel(defaults['learning_rate'])
       # ... training loop ...

   # This will cause a NameError
   print(defaults) # defaults is not accessible here

   train_model(0.01)
   ```

   To resolve this, either move the `defaults` declaration to a higher scope (e.g., global scope or an encompassing class) or pass it as an argument to the functions that require access. Alternatively, refactoring the code to better manage the scope and reduce nesting can also resolve this issue.

Let's illustrate these points with code examples.

**Example 1: Correct Import and Usage**

```python
# hyperparameters.py
hyperparameters = {
    'defaults': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10
    },
    'optimizer': 'AdamW'
}

# training_script.py
from hyperparameters import hyperparameters

learning_rate = hyperparameters['defaults']['learning_rate']
batch_size = hyperparameters['defaults']['batch_size']

#Rest of the PyTorch code using learning_rate and batch_size...

model = MyModel(learning_rate = learning_rate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


```

This example demonstrates the proper way to define and import a nested `defaults` dictionary, avoiding the `NameError`.


**Example 2: Incorrect Scope (Illustrating the Problem)**

```python
import torch

def create_model(lr):
    defaults = {'learning_rate': lr} # defaults defined inside a function
    model = torch.nn.Linear(10,2)
    optimizer = torch.optim.SGD(model.parameters(), lr=defaults['learning_rate'])
    return model, optimizer

model, optimizer = create_model(0.1)


#This will result in the NameError because defaults is not in the global namespace
print(defaults)
```

This highlights the `NameError` arising from attempting to access `defaults` outside its scope.


**Example 3: Resolving the Scope Issue**

```python
import torch

defaults = {'learning_rate': 0.01} # defaults defined globally

def create_model(lr):
    model = torch.nn.Linear(10,2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, optimizer


model, optimizer = create_model(defaults['learning_rate']) #Passing lr as an argument

print(defaults['learning_rate']) # Accessing defaults correctly here
```

This corrected version demonstrates passing `defaults` to `create_model`, resolving the scope issue.  The code accesses `defaults` successfully, as it's now in the global scope.


To further enhance your understanding of Python's scoping rules, I recommend consulting resources such as the official Python documentation on namespaces and variable scope, and a comprehensive Python tutorial covering the LEGB rule (Local, Enclosing function locals, Global, Built-in).  Understanding these concepts is fundamental to avoiding this and similar errors in Python and PyTorch development.  Furthermore, exploring documentation related to object-oriented programming in Python will clarify the concepts of class and instance variables and their respective scopes.
