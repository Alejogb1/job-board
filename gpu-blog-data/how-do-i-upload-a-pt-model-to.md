---
title: "How do I upload a .pt model to the Hugging Face Hub?"
date: "2025-01-30"
id: "how-do-i-upload-a-pt-model-to"
---
I've spent a considerable amount of time managing machine learning model deployments, and the nuances of uploading PyTorch models (.pt) to the Hugging Face Hub are crucial for collaborative and reproducible research. It's not as straightforward as just dragging and dropping; it involves careful consideration of model structure, repository management, and efficient version control.

The core task revolves around utilizing the `huggingface_hub` library, which provides a programmatic interface to interact with the Hugging Face Hub. This library facilitates not only uploading models but also managing repositories, datasets, and other resources. Before diving into code, it's essential to understand the typical workflow: 1) You must have a Hugging Face account and an API token. 2) Your model (.pt file) should ideally be accompanied by configuration files (such as `config.json`) and other relevant metadata, which are crucial for properly loading the model later. 3) You'll need to create or select a suitable repository on the Hub.

Let's walk through a step-by-step process, illustrated with concrete examples. The initial hurdle is authentication. The `huggingface_hub` library handles this transparently using your API token.

**Example 1: Basic Model Upload**

```python
from huggingface_hub import HfApi, hf_hub_login
import torch
import os

# Assuming you have a model.pt file and want to upload it
# Store the path to your model
model_path = "path/to/your/model.pt"
# Store the path to your model config
config_path = "path/to/your/config.json"
# Desired repository name on the Hub
repo_id = "your_username/your_model_name"

# Check if the token is set as environment variable
token_env = os.getenv("HF_TOKEN")
if token_env:
    hf_hub_login(token=token_env) # Using token from env
else:
    hf_hub_login() # Prompts for token if not in env

api = HfApi()

# Uploading only the model weight file
# This method also creates the repo if it does not exist
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="pytorch_model.bin",
    repo_id=repo_id,
    repo_type="model",
)

# If your model requires a specific config it can be uploaded here:
api.upload_file(
    path_or_fileobj=config_path,
    path_in_repo="config.json",
    repo_id=repo_id,
    repo_type="model",
)
print(f"Model uploaded to: https://huggingface.co/{repo_id}")

```

This first example demonstrates the fundamental process. I prefer to use `HfApi` to manage the upload. The `hf_hub_login()` method authenticates your session. If your API token is stored in the `HF_TOKEN` environment variable, it'll use that directly; otherwise, it'll prompt you. The crucial part is `api.upload_file()`. The `path_or_fileobj` parameter points to your .pt file and `path_in_repo` specifies the name it will have in the repository. Notice I've renamed the .pt file to `pytorch_model.bin` which is a convention within the Hugging Face ecosystem, but is not strictly required. The `repo_id` is your target repository. The `repo_type="model"` parameter specifies this is a model repository. If you have a configuration file for the model, like a `config.json` file, you should upload it similarly. These config files are important for properly reconstructing the model architecture when loading it.

However, real-world applications usually require more intricate management, including version control and custom model classes.

**Example 2: Model Upload with Version Control**

```python
from huggingface_hub import HfApi, hf_hub_login, create_repo
import torch
import os
from pathlib import Path

# Define Model Class
class MyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an example model and save it locally
input_size = 784
hidden_size = 128
output_size = 10
model = MyModel(input_size, hidden_size, output_size)
model_path = "my_model.pt"
torch.save(model.state_dict(), model_path)


# Define repo ID
repo_id = "your_username/my_custom_model"
# Check if the token is set as environment variable
token_env = os.getenv("HF_TOKEN")
if token_env:
    hf_hub_login(token=token_env) # Using token from env
else:
    hf_hub_login() # Prompts for token if not in env
api = HfApi()

# Create repo (if it does not exist)
try:
  create_repo(repo_id, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Could not create {repo_id}: {e} \n Ensure you have proper permissions.")


# Create a local directory representing the model structure
local_model_dir = Path("my_local_model")
local_model_dir.mkdir(exist_ok=True)


# Save a simplified config.json to the local model directory
config = {"input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size}

import json
with open(local_model_dir / "config.json", "w") as f:
    json.dump(config, f)

# Copy the .pt file to the local directory
import shutil
shutil.copy(model_path, local_model_dir / "pytorch_model.bin")

# Upload the entire local directory to the Hub
api.upload_folder(
    repo_id=repo_id,
    folder_path=local_model_dir,
    repo_type="model",
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
# Clean up created directories and files
shutil.rmtree(local_model_dir)
os.remove(model_path)

```

In this example, I've gone a step further. I've defined a custom PyTorch model class, `MyModel`, and saved its state dictionary. Instead of just uploading the .pt file, I simulate a more structured repository: I created a `local_model_dir`, added a `config.json`, and then use `api.upload_folder()` to push the entire directory to the Hub. This method makes it easier to manage model versions and associated files. Note the use of `try ... except` block, this is a common way to manage the creation of a repository. `create_repo()` with `exist_ok=True` will avoid errors if a repository already exists. Furthermore, I've included a cleanup at the end to delete all of the files and folders created during the model saving and upload processes.

The most important aspect of version control is not just uploading files but also tracking changes over time. For this, the `git` interface within the Hugging Face library is very useful.

**Example 3: Model Upload with Git Integration**

```python
from huggingface_hub import HfApi, hf_hub_login, create_repo
from huggingface_hub.utils import EntryNotFoundError
import torch
import os
from pathlib import Path
from git import Repo
import shutil


# Define Model Class
class MyModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an example model and save it locally
input_size = 784
hidden_size = 128
output_size = 10
model = MyModel(input_size, hidden_size, output_size)
model_path = "my_model.pt"
torch.save(model.state_dict(), model_path)


# Define repo ID
repo_id = "your_username/my_git_model"
# Check if the token is set as environment variable
token_env = os.getenv("HF_TOKEN")
if token_env:
    hf_hub_login(token=token_env) # Using token from env
else:
    hf_hub_login() # Prompts for token if not in env

api = HfApi()

# Create a local directory representing the model structure
local_model_dir = Path("my_local_model")
local_model_dir.mkdir(exist_ok=True)

# Save a simplified config.json to the local model directory
config = {"input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size}

import json
with open(local_model_dir / "config.json", "w") as f:
    json.dump(config, f)


# Copy the .pt file to the local directory
shutil.copy(model_path, local_model_dir / "pytorch_model.bin")

# Initialize Git Repository and add all files
try:
    repo = Repo.init(local_model_dir)
except Exception as e:
    print(f"Error Initializing git in {local_model_dir}: {e}")


repo.git.add(".")
repo.index.commit("Initial model upload")

# Attempt to clone the remote repo. If not found create it.
try:
    api.clone_repo(repo_id, local_dir=local_model_dir)
except EntryNotFoundError:
    create_repo(repo_id, repo_type="model")
    api.clone_repo(repo_id, local_dir=local_model_dir)



# Push local changes to the Hub
try:
    repo.git.push()
except Exception as e:
    print(f"Error pushing to hub: {e}")

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
# Clean up created directories and files
shutil.rmtree(local_model_dir)
os.remove(model_path)
```

This final example uses the `git` python library to manage the model repository. After creating the local model directory, files and git repository we now use `api.clone_repo` to clone the model repository from the Hub if it exists. If it does not it will be created and then cloned. Finally we push the local changes to the Hub. This enables proper version control allowing for tracking of changes in model weights and configurations over time.

In my experience, these three examples cover the most common scenarios when uploading PyTorch models to the Hugging Face Hub. Effective model management goes beyond just the upload; it also involves a robust model architecture definition, proper documentation, and responsible release practices.

For further learning, I recommend familiarizing yourself with the Hugging Face Hub's official documentation and also the documentation for the `huggingface_hub` library. Exploring tutorials on PyTorch and Git integration will also prove beneficial. Examining the structure of existing model repositories on the Hugging Face Hub is a great way to gain further insight.
