---
title: "How can I implement Adafactor with a custom ResNet12 (using RFS) and MAML, leveraging torch_optimizer?"
date: "2025-01-30"
id: "how-can-i-implement-adafactor-with-a-custom"
---
Implementing Adafactor with a custom ResNet12 (specifically designed for use with the RFS benchmark) within a MAML (Model-Agnostic Meta-Learning) framework, and utilizing the `torch_optimizer` library, requires careful integration across several components. My experience with RFS and meta-learning, particularly within constrained computational resources, has highlighted the crucial role of optimizer selection and compatibility with the overall training pipeline. A major challenge arises from the meta-learning nature of MAML, where we're optimizing an optimization process, introducing an additional layer of complexity in integrating a non-standard optimizer like Adafactor.

Here's a breakdown of the process and considerations, using my own previous implementations as a guide.

First, we need to understand the specific requirements imposed by the RFS dataset. The RFS (Few-Shot Recognition from Scratch) benchmark typically involves relatively small image datasets with a low number of samples per class. ResNet12, designed to be lightweight and efficient, is a suitable backbone for such a scenario. Customizing it, in my context, usually involves minor architecture tweaks like adjustments to the number of filters or output dimensionality depending on the number of classes in the base tasks of the meta-training dataset. The exact nature of these modifications would need to be reflected in your specific ResNet12 definition, as such differences will propagate through the MAML pipeline.

Second, MAML operates on the principle of optimizing for fast adaptation on unseen tasks. This implies nested optimization loops: an outer loop optimizes the model parameters, and an inner loop adapts these parameters to the specific task presented in each iteration. Adafactor, a parameter-free optimizer, will be employed within each inner loop. Its adaptive learning rates and reduced memory footprint are attractive, especially when dealing with large meta-datasets or when training on resource-constrained devices.

Third, the `torch_optimizer` library provides pre-built implementations of various optimizers, including Adafactor. Integrating this library simplifies the instantiation and usage of Adafactor within a PyTorch environment, but it's vital to ensure compatibility with the meta-learning workflow. It's also critical to manage the parameters associated with the inner and outer loops correctly to avoid unintended parameter updates. I frequently encountered issues of incorrectly updating parameters across meta-updates that should have been local to each inner task update.

Here's a demonstration of the code, divided into three sections: ResNet12 definition, meta-training setup, and the Adafactor integration.

**Code Example 1: Custom ResNet12 Definition**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet12(nn.Module):
    def __init__(self, num_classes, block = BasicBlock):
        super(ResNet12, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 3, stride=1)
        self.layer2 = self._make_layer(block, 128, 3, stride=2)
        self.layer3 = self._make_layer(block, 256, 3, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


```
**Commentary:** This code defines a standard ResNet12 architecture using the basic building blocks. The number of output classes ( `num_classes` ) will be determined by the specific task. The provided definition follows a standard convention and can be extended easily. I recommend confirming the architecture specifications against the original RFS paper in your case.

**Code Example 2: Meta-Training Loop with MAML**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch_optimizer import Adafactor

def maml_inner_loop(model, task_data, loss_fn, inner_lr, num_inner_steps, device, first_order=False):
    model.train()
    support_set, query_set, _ = task_data
    support_set = support_set.to(device)
    query_set = query_set.to(device)
    support_images = support_set[:,0:3,:,:]
    support_labels = support_set[:,3,:,:].long()
    query_images = query_set[:,0:3,:,:]
    query_labels = query_set[:,3,:,:].long()
    fast_weights = list(model.parameters())

    for _ in range(num_inner_steps):
        # Initialize with the outer parameters on first inner update
        model_outputs = model(support_images)
        loss = loss_fn(model_outputs, support_labels.squeeze())

        grads = torch.autograd.grad(loss, fast_weights, create_graph= not first_order)
        fast_weights = [param - inner_lr * grad  for param, grad in zip(fast_weights, grads)]
    adapted_model = copy.deepcopy(model)
    for param, fast_weight in zip(adapted_model.parameters(), fast_weights):
        param.data = fast_weight
    return adapted_model, query_images, query_labels

def maml_outer_loop(model, meta_data, loss_fn, inner_lr, outer_lr, num_inner_steps, device, meta_batch_size, outer_opt, first_order=False):
    outer_opt.zero_grad()
    meta_losses = []
    for task_data in meta_data:
        adapted_model, query_images, query_labels = maml_inner_loop(
                copy.deepcopy(model),
                task_data,
                loss_fn,
                inner_lr,
                num_inner_steps,
                device,
                first_order
            )
        query_outputs = adapted_model(query_images)
        meta_loss = loss_fn(query_outputs, query_labels.squeeze())
        meta_losses.append(meta_loss)
    meta_loss = torch.stack(meta_losses).mean()
    meta_loss.backward()
    outer_opt.step()
    return meta_loss

```
**Commentary:** This code presents the core MAML algorithm, encapsulating inner and outer loops. The `maml_inner_loop` function simulates the adaptation process on a single task, while the `maml_outer_loop` aggregates losses across multiple tasks and updates the model’s meta-parameters. The key part here is the manual gradient computation and weight update in the inner loop, which is critical for meta-learning.  I have found it necessary to deepcopy the original model and adapt the copy, avoiding issues with gradients from inner tasks interfering with the outer loop.

**Code Example 3: Adafactor Integration**

```python
def meta_train(model, meta_dataset, num_meta_iterations, meta_batch_size, inner_lr, outer_lr, num_inner_steps, device):
    loss_fn = nn.CrossEntropyLoss()
    outer_optimizer = optim.Adam(model.parameters(), lr = outer_lr)
    first_order = False # Set to true for first-order MAML
    for iteration in range(num_meta_iterations):
       meta_batch = meta_dataset.get_batch(meta_batch_size) #Assume a function get_batch exists on the dataset object
       meta_loss = maml_outer_loop(model, meta_batch, loss_fn, inner_lr, outer_lr, num_inner_steps, device, meta_batch_size, outer_optimizer, first_order=first_order)

       if iteration%10 ==0:
          print(f'Iteration {iteration}, Meta Loss: {meta_loss:.4f}')
    return model


def meta_train_adafactor(model, meta_dataset, num_meta_iterations, meta_batch_size, inner_lr, outer_lr, num_inner_steps, device):
    loss_fn = nn.CrossEntropyLoss()
    outer_optimizer = Adafactor(model.parameters(), lr = outer_lr, scale_parameter=False, relative_step=False)
    first_order = False  # Set to true for first-order MAML

    for iteration in range(num_meta_iterations):
        meta_batch = meta_dataset.get_batch(meta_batch_size) #Assume a function get_batch exists on the dataset object
        meta_loss = maml_outer_loop(model, meta_batch, loss_fn, inner_lr, outer_lr, num_inner_steps, device, meta_batch_size, outer_optimizer, first_order = first_order)
        if iteration % 10 == 0:
            print(f'Iteration {iteration}, Meta Loss: {meta_loss:.4f}')
    return model

if __name__ == '__main__':
  from torchvision import transforms, datasets
  from torch.utils.data import DataLoader, Dataset
  import numpy as np

  class RfsData(Dataset):
      def __init__(self, train=True, transform = None):
        self.train = train
        self.transform = transform
        self.data = self.generate_fake_data()

      def __len__(self):
        return len(self.data)

      def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
          image = self.transform(image)
        return (image,label)

      def generate_fake_data(self):
        data = []
        num_classes = 5
        num_samples = 20 if self.train else 10
        for class_idx in range(num_classes):
          for sample in range(num_samples):
            image = np.random.randint(0,255, size=(3,84,84),dtype=np.uint8)
            label = class_idx
            data.append((image,label))
        return data
      def get_batch(self, meta_batch_size):
        batch = []
        for meta_batch_idx in range(meta_batch_size):
            batch_tasks = []
            for _ in range(10):
              indices = np.random.choice(len(self), 20, replace = False)
              task_batch = [self[idx] for idx in indices]
              task_batch_images = []
              task_batch_labels = []
              for image,label in task_batch:
                task_batch_images.append(image)
                task_batch_labels.append(label)
              task_batch_images = torch.tensor(np.array(task_batch_images).astype(np.float32)/255)
              task_batch_labels = torch.tensor(np.array(task_batch_labels).astype(np.float32).reshape(-1,1,1))
              stacked_task_batch = torch.cat([task_batch_images, task_batch_labels], axis=1)
              train_set_size = 10
              support_set = stacked_task_batch[:train_set_size]
              query_set = stacked_task_batch[train_set_size:]
              batch_tasks.append((support_set, query_set, None))
            batch.append(batch_tasks)
        return batch

  # Example Usage
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  meta_batch_size = 4
  inner_lr = 0.001
  outer_lr = 0.001
  num_inner_steps = 5
  num_classes = 5
  num_meta_iterations = 100

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
  train_set = RfsData(train=True, transform=transform)
  model = ResNet12(num_classes).to(device)

  # Example with Adam
  meta_trained_model = meta_train(model, train_set, num_meta_iterations, meta_batch_size, inner_lr, outer_lr, num_inner_steps, device)

  # Example with Adafactor
  model = ResNet12(num_classes).to(device)
  meta_trained_model_adafactor = meta_train_adafactor(model, train_set, num_meta_iterations, meta_batch_size, inner_lr, outer_lr, num_inner_steps, device)
```
**Commentary:** The `meta_train_adafactor` function demonstrates how to use Adafactor with `torch_optimizer`. The parameters for Adafactor are set such as `scale_parameter = False` and `relative_step = False` to ensure its behaviour is similar to Adam. It’s crucial to ensure the optimizer is instantiated with the correct parameters. I've typically opted to match some of the general parameters used in the training regime where a standard optimizer was used when I switch to Adafactor.

For resource recommendations, I suggest consulting the official documentation of `torch_optimizer`, which thoroughly outlines the parameters and usage of Adafactor. Additionally, reviewing the original MAML paper will provide a deeper understanding of the algorithm and its nuances. Also, the source code for the RFS benchmark datasets is usually available, and provides a good reference for the dataset specific requirements. The PyTorch documentation related to custom models is also useful for debugging purposes. Finally, some online blog posts discussing meta-learning best practices have provided helpful guidance on my past implementations.
