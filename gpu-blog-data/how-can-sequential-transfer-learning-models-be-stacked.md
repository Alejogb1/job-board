---
title: "How can sequential transfer learning models be stacked?"
date: "2025-01-30"
id: "how-can-sequential-transfer-learning-models-be-stacked"
---
Sequential transfer learning, where knowledge gained from one task is applied to a subsequent related task, presents a unique challenge when stacking models. Simply feeding the output of one model directly into another, as might be done in a traditional model ensemble, does not effectively leverage the sequential nature of the learning process. Instead, a careful consideration of feature representation and model adaptation is necessary. I have encountered this specific problem in multiple projects, ranging from fine-tuning language models for downstream tasks to adapting object detection models for progressively complex scenes, and a structured methodology has proven to be consistently beneficial.

The core issue lies in the changing data distributions and the feature spaces between different stages of transfer learning. When a model is trained on an initial, source task, its internal feature representation becomes optimized for that specific task. If a subsequent model, in the target task, operates on the raw output or even penultimate layer activations of the source model without modification, the new model may struggle to extract meaningful information. This is due to potential mismatches between the source's feature space and the optimal feature space for the target task. Therefore, stacking involves not only chaining models together, but also adapting the source model’s output to be relevant for the target task. This adaption can take various forms, including feature extraction, transformation, or specialized input layers.

My typical approach involves the following: First, I identify the most relevant output from the source model for the target task. This is not always the final output layer’s predictions. Often, higher-level representations from earlier layers offer a more generalized understanding that can be effectively fine-tuned for the new task. For example, in natural language processing, the intermediate layer of a transformer model may be more useful than the output logits when adapting to a more specific task. Second, I introduce a trainable adaptation layer or a series of layers to map the source model's output into a more appropriate feature space for the target model. This can include fully connected layers, pooling layers, or more specialized architectures depending on the type of data and model. Third, the target model is trained on the transformed output of the adapted source model, while the source model's parameters are typically frozen or fine-tuned at a very low learning rate, preventing catastrophic forgetting of previously learned features.

Let’s consider a specific case: using a pre-trained image classification model as the foundation for a new object detection task. Suppose we’ve already trained a ResNet-50 model on ImageNet (source model) and need to adapt it to detect cars in specific city scenes (target task). The classification output of ResNet-50 is not directly suitable for this. Instead, we can use the feature maps before the final classification layer as our base.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50'):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.features.parameters():
            param.requires_grad = False # Freeze pretrained model

    def forward(self, x):
        return self.features(x)

class AdaptationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptationLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ObjectDetector(nn.Module):
    def __init__(self, feature_channels=2048, adaptation_channels=256, num_classes=2):
        super(ObjectDetector, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.adaptation_layer = AdaptationLayer(feature_channels, adaptation_channels)
        self.bbox_predictor = nn.Conv2d(adaptation_channels, 4, kernel_size=1) # bounding box regressor
        self.class_predictor = nn.Conv2d(adaptation_channels, num_classes, kernel_size=1) # object classifier

    def forward(self, x):
        features = self.feature_extractor(x)
        adapted_features = self.adaptation_layer(features)
        bboxes = self.bbox_predictor(adapted_features)
        class_probs = self.class_predictor(adapted_features)
        return bboxes, class_probs


# Example instantiation
model = ObjectDetector()
input_tensor = torch.randn(1, 3, 224, 224)
bbox_output, class_output = model(input_tensor)
print("Bounding Box Output Shape:", bbox_output.shape)
print("Class Output Shape:", class_output.shape)

```
In this first example, the `FeatureExtractor` class loads a pretrained ResNet50 model and extracts its feature layers up to the average pooling stage, freezing their weights. The `AdaptationLayer` class then transforms this feature map, with a simple convolution and ReLU for demonstration, into a lower-dimensional representation. Finally, the `ObjectDetector` combines feature extraction and adaptation, adding two small convolutional networks for bounding box regression and class prediction, respectively.

A second example focuses on sequential transfer learning in NLP. Here, a pre-trained transformer model trained on a generic corpus can be adapted for sentiment analysis of social media text. The intermediate representations of the transformer offer more contextualized information than word embeddings alone.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(TransformerFeatureExtractor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        for param in self.transformer.parameters():
          param.requires_grad=False


    def forward(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.transformer(**inputs)
        return outputs.last_hidden_state[:, 0, :] # CLS token representations


class SentimentClassifier(nn.Module):
    def __init__(self, transformer_dim=768, hidden_dim=256, num_classes=3):
      super(SentimentClassifier,self).__init__()
      self.transformer_extractor=TransformerFeatureExtractor()
      self.fc1=nn.Linear(transformer_dim,hidden_dim)
      self.relu=nn.ReLU()
      self.fc2=nn.Linear(hidden_dim,num_classes)

    def forward(self, texts):
      features=self.transformer_extractor(texts)
      hidden = self.relu(self.fc1(features))
      logits = self.fc2(hidden)
      return logits
# Example instantiation
model = SentimentClassifier()
sample_texts = ["This movie is terrible", "I love this restaurant", "This is an okay place"]
logits = model(sample_texts)
print("Logits Output Shape:", logits.shape)

```
In this code, the `TransformerFeatureExtractor` leverages the Hugging Face `transformers` library to load a pre-trained BERT model. Instead of just the final logits, the class extracts the CLS token embedding from the last hidden layer. These embeddings, representing the contextualized meaning of the input, are passed through a couple of linear layers in the `SentimentClassifier`, allowing for fine-tuning for the sentiment analysis task, while maintaining the transformer model as frozen for feature extraction only.

A final example involves adapting a reinforcement learning model initially trained in a simple simulation environment to function in a more complex real-world environment. Here, rather than passing outputs directly, we adapt the state space representation used by the RL agent. Suppose an agent learns a basic navigation policy in a grid-world simulation. The simulation’s state space might be a simple coordinate representation, while the real-world environment’s state space might be more complex using images from a camera.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GridWorldState(nn.Module):
    def __init__(self, state_size=2, hidden_size=128):
      super(GridWorldState, self).__init__()
      self.fc = nn.Linear(state_size, hidden_size)
      self.relu = nn.ReLU()

    def forward(self, state):
       return self.relu(self.fc(torch.tensor(state, dtype=torch.float32)))


class RealWorldStateAdapter(nn.Module):
  def __init__(self, image_channels=3, image_size=64, hidden_size=128):
      super(RealWorldStateAdapter, self).__init__()
      self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1)
      self.relu1=nn.ReLU()
      self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
      self.relu2 = nn.ReLU()
      self.flatten = nn.Flatten()
      self.fc = nn.Linear(64 * (image_size//4)**2, hidden_size)
      self.relu3=nn.ReLU()

  def forward(self,image):
    x=self.relu1(self.conv1(image))
    x=self.relu2(self.conv2(x))
    x=self.flatten(x)
    return self.relu3(self.fc(x))

class RLPolicy(nn.Module):
  def __init__(self, state_size=128, action_size=4):
      super(RLPolicy,self).__init__()
      self.fc1=nn.Linear(state_size, 64)
      self.relu1=nn.ReLU()
      self.fc2=nn.Linear(64,action_size)

  def forward(self, state):
      x = self.relu1(self.fc1(state))
      return self.fc2(x)


# Example Usage (Illustrative)
grid_state_processor = GridWorldState()
real_world_adapter=RealWorldStateAdapter()
rl_policy = RLPolicy()


grid_state = [2, 3] #example location
adapted_grid_state=grid_state_processor(grid_state)
print("Grid State Output Shape:", adapted_grid_state.shape)
# Image processing for real-world states (Example: dummy image data)
dummy_image = torch.randn(1, 3, 64, 64)
adapted_real_state=real_world_adapter(dummy_image)
print("Real World State Output Shape:", adapted_real_state.shape)

# RL policy with states (Illustrative example, needs proper learning)
grid_actions=rl_policy(adapted_grid_state)
real_actions=rl_policy(adapted_real_state)
print("Grid Actions:", grid_actions)
print("Real Actions:",real_actions)

```
In this final example, we use a `GridWorldState` class to process simple location based states. This is not the focus of transfer learning. Instead we introduce `RealWorldStateAdapter` which takes raw image inputs and processes them via two CNN layers. The resulting output is of the same dimensions as the grid world state representation. Then, the policy from the first model (`RLPolicy`), trained using the grid world state output as an input can be reused with the real-world output, effectively transferring the learned policy to the new environment after adapting the states. This example is intended to illustrate how the state space itself needs adapting, as opposed to the model's output in other examples.

For resources to improve understanding of this type of modeling, I would recommend exploring texts covering deep learning model design, specifically focusing on transfer learning and feature engineering. Resources covering modern neural network architectures such as convolutional networks, recurrent networks and transformers are also relevant. Books covering reinforcement learning techniques may also be valuable if the goal is to leverage models in a reinforcement setting, as well as literature on specific application domains that may involve sequential transfer learning. These concepts are often discussed in research articles focusing on specific fields, such as computer vision, natural language processing, and robotics.
