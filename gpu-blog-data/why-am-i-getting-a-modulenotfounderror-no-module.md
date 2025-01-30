---
title: "Why am I getting a ModuleNotFoundError: No module named '__builtin__' when using load_learner() in FastAI?"
date: "2025-01-30"
id: "why-am-i-getting-a-modulenotfounderror-no-module"
---
The `ModuleNotFoundError: No module named '__builtin__'` encountered when utilizing `load_learner()` within the FastAI library stems from an incompatibility between the Python version employed and the saved model's environment.  Specifically, the `__builtin__` module, prevalent in Python 2, has been renamed `builtins` in Python 3.  A model trained and saved under a Python 2 environment will inherently reference `__builtin__`, causing this error when loaded within a Python 3 environment which lacks this module.  This incompatibility arises frequently during collaborative projects or when loading pre-trained models from diverse sources. My experience debugging similar issues in large-scale image classification projects solidified my understanding of this nuance.

**1. Explanation:**

The FastAI library, while designed for ease of use, relies heavily on the underlying Python environment's capabilities.  During the model training process, FastAI serializes the model's architecture, weights, and importantly, the environment dependencies.  This serialization process captures the specific modules and versions present during training. If the training environment uses Python 2, and its associated `__builtin__` module, the saved model will retain this dependency. When attempting to load this model using `load_learner()` within a Python 3 environment, the interpreter attempts to locate `__builtin__`, which is absent, resulting in the `ModuleNotFoundError`.

This is not merely a cosmetic issue; the error indicates a fundamental mismatch between the runtime environment and the model's saved dependencies.  Ignoring this incompatibility will lead to further errors, potentially corrupting the model's functionality or causing unexpected behaviors.  Solving this requires creating a consistent environment between model training and loading.


**2. Code Examples with Commentary:**

**Example 1:  Illustrating the Problem:**

```python
# Assume 'model.pkl' was trained using Python 2.7

from fastai.vision.all import *

try:
    learn = load_learner('model.pkl')
    print("Model loaded successfully.")  # This line will likely not be reached.
except ModuleNotFoundError as e:
    print(f"Error loading model: {e}")
```

This example demonstrates the typical scenario. The `load_learner()` function attempts to load the model. If the model's saved environment contained `__builtin__`, the exception will be raised, halting execution. The `try-except` block gracefully handles the error, providing informative feedback.


**Example 2: Creating a Compatible Environment (using `virtualenv`):**

```bash
# Create a virtual environment (replace 'myenv' with your desired name)
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install required packages (replace with your specific requirements)
pip install fastai torch torchvision

# Train your model within this environment (ensuring Python 3 compatibility)
# ... your FastAI training code ...

# Save your model within this environment

# Loading the model (within the same activated environment)
python your_script.py
```

This example highlights the crucial step of environment management. Using a virtual environment ensures consistent dependencies between training and loading.  By activating the same virtual environment where the model was trained, the necessary packages and Python version are guaranteed, preventing the `ModuleNotFoundError`. The crucial part is ensuring the model is trained and loaded within the same, carefully managed environment.


**Example 3:  Addressing Incompatibility using `pickle` (Advanced and Risky):**

```python
import pickle
import sys

from fastai.vision.all import *


try:
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)


    #Attempt to manually patch the model - HIGHLY DISCOURAGED
    if "__builtin__" in str(model_data): #dangerous and inaccurate check
        for key, value in model_data.items():
          if '__builtin__' in str(value):
            try:
              model_data[key] = value.replace('__builtin__', 'builtins')
            except:
              print("Patching failed")
    
    learn = load_learner(model_data) # This might still fail

    print("Model loaded successfully (after patching).")

except Exception as e:
    print(f"Error loading model: {e}")

```

This example demonstrates a more involved, but highly discouraged, approach. Manually attempting to modify the loaded model data to replace instances of `__builtin__` with `builtins` is extremely fragile.  The success of this method depends entirely on the structure and serialization of the specific model, and is highly susceptible to breaking the model's functionality. This approach should be considered a last resort only if other solutions fail, understanding the high risk of corrupting the model. I personally recommend against this approach as a routine solution, only resorting to such techniques when dealing with truly legacy, non-editable models.  In most cases, the recommended approach is to retrain your model in a compatible environment.


**3. Resource Recommendations:**

For a deeper understanding of Python environments and virtual environments, consult the official Python documentation on the `venv` module.  Familiarize yourself with the FastAI documentation on model saving and loading, particularly regarding best practices for environment management. The comprehensive documentation for the `pickle` module is recommended for understanding its capabilities and limitations, particularly in handling complex objects like trained machine learning models.  Understanding these core concepts is paramount to avoiding future incompatibility issues.  Furthermore, reviewing tutorials on deploying machine learning models can provide additional insight into best practices for maintaining consistent runtime environments across different phases of model development and deployment.
