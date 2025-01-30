---
title: "How can I load a PyTorch transformer model from a checkpoint without loading pre-trained weights?"
date: "2025-01-30"
id: "how-can-i-load-a-pytorch-transformer-model"
---
The crux of loading a PyTorch transformer model from a checkpoint without its pre-trained weights lies in meticulously controlling the state dictionary during the loading process.  Typically, `torch.load()` coupled with `model.load_state_dict()` will overwrite a model's existing parameters, including its randomly initialized ones, with those stored in the checkpoint. To circumvent this, we must selectively load only model architecture data and avoid restoring the parameter tensors from the checkpoint. This is primarily achieved by initializing a model from the config, then loading only the parameters that matches the *structure* of the architecture but ignoring the actual *data* of the checkpoint, allowing us to start from scratch with random weights.

The usual workflow, which I have encountered numerous times when fine-tuning models for specific tasks, involves a checkpoint generated after initial pre-training on a large corpus. This checkpoint contains both the model's architecture specification (layers, embedding dimensions, attention heads, etc.) *and* the learned weights.  However, if one wants to evaluate architecture changes or begin fresh training on a different dataset, the pre-trained weights are not desirable. The conventional `model.load_state_dict()` operation would directly overwrite the newly initialized model with the pre-trained state, preventing this type of controlled initialization. We can instead leverage the `state_dict()` and the initialization of the model using `from_pretrained` method to get an empty template to overwrite. The key idea is that by initializing a new model with the same configuration as the model whose weights we want to replace, we create the right shape for parameters, then we load the old model's state dictionary without replacing the current state of the parameters.

Let's break down the process with concrete examples. Assume we have a transformer model, perhaps derived from the Hugging Face library, though the principle applies generally to any PyTorch model. Here is an example of loading a pre-trained model from a checkpoint.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Assume 'path/to/checkpoint.pth' is our saved checkpoint
checkpoint_path = "path/to/checkpoint.pth"

# Load the checkpoint which includes both weights and architecture
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
loaded_model = checkpoint['model']

# This line will overwrite the model's weights
# loaded_model.load_state_dict(checkpoint['model_state_dict'])
```

In this first example, assuming the checkpoint file contains a key called `'model'` this contains the model object directly, and `'model_state_dict'` which is the saved state dictionary which contains the weights of the model. We would load them as normal by passing the loaded state dictionary to `loaded_model.load_state_dict()`, which overwrites the new model with the checkpoint weights and architecture. If we had initialized a model ourselves, this code would have replaced it with the loaded checkpoint. The point is that the line I have commented out is the usual process of loading weights. The code below gives an example of what happens if we initialize a new model from the config and then load the weights.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Assume 'path/to/checkpoint.pth' is our saved checkpoint
checkpoint_path = "path/to/checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
loaded_model = checkpoint['model'] # This has the config info

# Extract the configuration
config = loaded_model.config

# Initialize a new model from the configuration, which gives us the correct architecture
new_model = AutoModelForSequenceClassification.from_config(config)

# Load the state dict from the checkpoint
state_dict_from_checkpoint = loaded_model.state_dict()

#Load only the weights into the new model, using the checkpoint's state dict's keys
new_model.load_state_dict(state_dict_from_checkpoint, strict=False)
```

In this second example, we first load the checkpoint, and extract the configuration from the model object. We then initialize a new model using this config. This is where the trick happens. Instead of loading the weights into the loaded model (which has the pre-trained weights), we initialize a new model, `new_model`, which has the correct architecture. We then extract the state dictionary from the *checkpoint* which we then load into the *new model*. We use strict=False because when loading the state dictionary we are only loading the weight parameters and not the model class's attributes. This means that the new model has the same architecture and weights as the checkpoint, but is not an instance of the original checkpoint.

Finally, if we only have the `state_dict`, and the model class to load from, but not the full model object, we need to load a config file separately, as in the following example.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

# Assume 'path/to/checkpoint.pth' is our saved checkpoint, which includes ONLY the state_dict
checkpoint_path = "path/to/checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
state_dict_from_checkpoint = checkpoint['model_state_dict']

# Assume 'path/to/config.json' is the config file related to the checkpoint
config_path = 'path/to/config.json'
config = AutoConfig.from_pretrained(config_path)

# Initialize a new model from the configuration, which gives us the correct architecture
new_model = AutoModelForSequenceClassification.from_config(config)

#Load only the weights into the new model, using the checkpoint's state dict's keys
new_model.load_state_dict(state_dict_from_checkpoint, strict=False)

```

In this last example, we load a state dictionary, and a configuration file, then initialize the model from that configuration, and load the weights into the new model, preserving its weights as initialized by default.

Throughout these examples, the critical aspect is the `strict=False` argument passed to `load_state_dict`. When `strict` is set to `True` (which is the default), PyTorch requires that *every* key in the loaded `state_dict` match a key in the target model. With `strict=False`, mismatched keys are ignored. In our case, we typically might have the full model (with architecture and weights) stored in the checkpoint, whereas we only want the weights, which can be missing some of the state information in the model's class. It allows us to load the parameters and ignore the other keys. In my experience, this has proved to be necessary when loading only the state dict into a pre-created model of a given class. This approach ensures that the model is initialized with randomly-initialized weights while inheriting the architecture from the saved checkpoint.

To reiterate, the key is to avoid the direct loading of the full `state_dict` into the pre-trained model instance as this would overwrite any changes you made to the weights, and overwrite your newly instantiated model. Instead, extract only the architecture information to instantiate a *new* model, then load the weights from the checkpoint into the new model, which you control.

For more depth and clarity, the PyTorch documentation on saving and loading models, specifically the `torch.save()` and `torch.load()` functions and the `load_state_dict()` method are crucial references. Additionally, the Hugging Face Transformers library documentation provides details on accessing model configurations and state dictionaries when working with pre-trained transformer models. The documentation for a specific model class will often give specifics on how to initialize and load models from configuration, or `state_dicts`. Research papers and blog posts that detail fine-tuning specific model architectures often provide implementation details on checkpoint loading and configuration management, which are relevant.
