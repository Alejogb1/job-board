---
title: "How can TCAV be used in PyTorch?"
date: "2025-01-30"
id: "how-can-tcav-be-used-in-pytorch"
---
Tensor Concept Activation Vectors (TCAV), initially developed for TensorFlow, offers a method to interpret the internal representations of deep neural networks by identifying directions in the model's activation space that correspond to high-level concepts. While not directly built into PyTorch, we can implement it using PyTorch's flexibility and automatic differentiation engine. I've personally used a similar adaptation across several projects investigating model biases and interpretable feature extraction. The core idea involves defining a concept as a set of example inputs, training a classifier to distinguish between instances of that concept and a set of random examples, and then using the learned classifier's weight vector to quantify concept sensitivity.

The primary steps to implement TCAV in PyTorch are as follows:

1.  **Data Collection:** Gather two distinct sets of data: example inputs representing the target concept (concept examples) and a set of random or unrelated inputs (random examples). The concept examples should be diverse and representative of the desired high-level concept.
2.  **Activation Extraction:** Obtain the internal layer activations of the neural network for both the concept and random examples. This often involves selecting an intermediate layer that you believe will capture meaningful representations. We will then aggregate the individual activations into a single representation per image.
3.  **Concept Classifier Training:** Train a linear classifier (e.g., logistic regression) to distinguish between the aggregated activations of concept examples and random examples. The learned weights of this classifier will form the concept activation vector (CAV).
4.  **TCAV Score Calculation:** Given a new input, extract the corresponding activations, project these activations onto the CAV, and compute the directional derivative to quantify concept sensitivity. This directional derivative serves as the TCAV score.

Let’s look at some code examples. The first will handle activation extraction:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def extract_activations(model, layer_name, data_loader, device):
    """
    Extracts activations from a specific layer of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.
        layer_name (str): Name of the layer from which to extract activations
        data_loader (DataLoader): DataLoader for input data
        device (str): Device for computation ('cuda' or 'cpu')

    Returns:
        torch.Tensor: A tensor containing the extracted activations
    """
    model.eval() # Set model to eval mode
    activations = []

    def hook(module, input, output):
      activations.append(output.detach().cpu().numpy()) # captures activations

    module = dict(model.named_modules())[layer_name] # finds specific layer by name
    handle = module.register_forward_hook(hook)

    with torch.no_grad(): # disables gradient calculation
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            model(inputs)

    handle.remove()
    return torch.from_numpy(np.concatenate(activations)).float()
```

This Python function utilizes a forward hook to capture intermediate layer activations. The `extract_activations` function accepts a PyTorch model, the target layer name, and a DataLoader as inputs. The forward hook will save the activations after each pass in inference mode and then concatenate them into a single tensor. The crucial part involves using the `register_forward_hook` method, which allows us to insert a custom function that intercepts the output of a layer. We then disable gradient computation to ensure the process is efficient and we detach to copy the activation data to cpu. For more efficient memory management you can batch the activation saving, but for simplicity it is done at the end.

Now, the second code example demonstrates training the concept classifier and calculating the TCAV score:

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_concept_classifier(concept_activations, random_activations, C=1.0, max_iter=1000):
  """
  Trains a linear classifier to distinguish between concept and random activations.

  Args:
      concept_activations (torch.Tensor): Activations for concept examples
      random_activations (torch.Tensor): Activations for random examples
      C (float): Regularization strength
      max_iter (int): Maximum iterations

  Returns:
      sklearn.linear_model.LogisticRegression: Trained classifier
  """
  X = torch.cat((concept_activations, random_activations), dim=0).numpy()
  y = np.concatenate((np.ones(len(concept_activations)), np.zeros(len(random_activations))))

  classifier = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear')
  classifier.fit(X, y)
  return classifier

def calculate_tcav_score(classifier, input_activations):
  """
  Calculates the TCAV score for a given input using the trained classifier.

  Args:
      classifier (sklearn.linear_model.LogisticRegression): Trained classifier
      input_activations (torch.Tensor): Activations for a new input

  Returns:
      float: TCAV score for the input
  """
  input_activations = input_activations.numpy() # change to numpy for sklearn
  concept_vector = classifier.coef_.reshape(-1)
  score = np.dot(input_activations, concept_vector)
  return score
```

This Python code snippet involves training a logistic regression model using scikit-learn. The `train_concept_classifier` function concatenates the concept and random activations to form the training data and the corresponding target labels. The linear model will then fit the data and return the classifier. The `calculate_tcav_score` function computes the dot product of the new input activations with the learned classifier weights which represent the concept vector, giving the score. These functions provide a method to compute the TCAV score for individual inputs, using a linear model and extracted activations. Note that for other linear models, you can change out `LogisticRegression`.

The final code segment brings all the functions together and demonstrates how to apply them:

```python
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import glob

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0 # returning label 0 because we are only interested in inputs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(pretrained=True).to(device) # select a model
    layer_name = 'layer4.2.relu' # choose intermediate layer

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # setup fake data directories
    os.makedirs("concept", exist_ok=True)
    os.makedirs("random", exist_ok=True)
    os.makedirs("test", exist_ok=True)
    for i in range(50): # generate 50 fake concept images, random images, and test images.
      Image.new('RGB', (224, 224), color = (i*5%256,i*7%256,i*9%256)).save(f"concept/concept_{i}.png")
      Image.new('RGB', (224, 224), color = (i*3%256,i*4%256,i*6%256)).save(f"random/random_{i}.png")
      Image.new('RGB', (224, 224), color = (i*1%256,i*2%256,i*3%256)).save(f"test/test_{i}.png")

    concept_paths = sorted(glob.glob("concept/*.png")) # collect data
    random_paths = sorted(glob.glob("random/*.png"))
    test_paths = sorted(glob.glob("test/*.png"))

    concept_dataset = CustomDataset(concept_paths, transform=transform) # apply transformations
    random_dataset = CustomDataset(random_paths, transform=transform)
    test_dataset = CustomDataset(test_paths, transform=transform)

    concept_loader = DataLoader(concept_dataset, batch_size=10, shuffle=False)
    random_loader = DataLoader(random_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    concept_activations = extract_activations(model, layer_name, concept_loader, device) # extract activations
    random_activations = extract_activations(model, layer_name, random_loader, device)

    classifier = train_concept_classifier(concept_activations, random_activations) # train classifier

    for i, (inputs, _) in enumerate(test_loader):
      inputs = inputs.to(device)
      input_activations = extract_activations(model, layer_name,
              DataLoader(TensorDataset(inputs), batch_size=1), device) #extract activations for a single input

      tcav_score = calculate_tcav_score(classifier, input_activations) # calculate the tcav score
      print(f"TCAV score for example {i}: {tcav_score:.4f}")

if __name__ == "__main__":
    main()
```

This script demonstrates a full example of using the functions we discussed above. First, it sets up the environment and loads a pre-trained ResNet50 model. It also defines transformations to prepare input images for the model. The code generates fake image data for concept, random, and test sets, then uses custom datasets and dataloaders. It extracts the activations for the concept and random sets, trains a classifier, and computes TCAV scores for new test images. These scores indicate the degree to which the test images activate the learned concept.

For further exploration, several resources provide a strong background on model interpretability. Research papers on topics like "Activation Maximization," and "Gradient-weighted Class Activation Mapping" can further enrich your knowledge. Additionally, model interpretability sections of popular machine learning textbooks will provide a solid overview of several common approaches. Online resources published by well-known universities are also an important source of knowledge for both understanding and developing new approaches.
