---
title: "How to save and load PyTorch checkpoints?"
date: "2025-01-30"
id: "how-to-save-and-load-pytorch-checkpoints"
---
Saving and loading PyTorch checkpoints is crucial for managing model training, particularly in scenarios involving extensive datasets or computationally expensive models.  My experience working on large-scale natural language processing tasks has highlighted the importance of robust checkpointing strategies to prevent the loss of significant training progress.  The core concept revolves around serialization: transforming the model's internal state—weights, biases, optimizer parameters, and potentially other relevant data—into a persistent storage format, typically a file, and subsequently reconstituting that state from the file.  This process allows resuming training from a specific point, comparing model versions, or deploying a pre-trained model without retraining from scratch.


**1. Clear Explanation:**

PyTorch provides convenient mechanisms for checkpointing via the `torch.save()` and `torch.load()` functions.  `torch.save()` serializes Python objects, including PyTorch models and optimizers, to a file.  The serialization format is typically a pickle file (`.pth` or `.pt` extension is common), but other formats can be used by specifying custom serializers.  `torch.load()` performs the reverse operation, deserializing the saved object and restoring it to memory.  Crucially, understanding what data you are saving and loading is paramount.  Simply saving the model's state dictionary (`model.state_dict()`) will only preserve the model's parameters.  To resume training, you'll also need to save the optimizer's state dictionary (`optimizer.state_dict()`).  Further, saving additional metadata like the epoch number or training loss can significantly enhance reproducibility and streamline the workflow.


**2. Code Examples with Commentary:**


**Example 1: Saving and Loading a Model and Optimizer State**

This example demonstrates the most common approach, preserving both the model's parameters and the optimizer's state.  This is essential for resuming training exactly where it left off.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize model and optimizer
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training loop (replace with your actual training loop)
for epoch in range(10):
    # ... your training code here ...
    if (epoch + 1) % 5 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss  # Add your loss value here
        }
        torch.save(checkpoint, f'checkpoint_{epoch+1}.pth')

# Load the checkpoint
checkpoint = torch.load('checkpoint_5.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Resume training from the loaded checkpoint
# ... your training code here ...

```


**Example 2: Saving only the Model's State Dictionary**

This example focuses solely on saving the model's parameters.  This is suitable when you only need to deploy a pre-trained model without resuming training.  Note that loading in this case doesn't require an optimizer.

```python
import torch
import torch.nn as nn

# ... (Model definition as in Example 1) ...

# Initialize model
model = SimpleModel()

# ... your training code here ...

# Save only the model's state_dict
torch.save(model.state_dict(), 'model_weights.pth')

# Load the model's state_dict
model = SimpleModel() # Instantiate a new model
model.load_state_dict(torch.load('model_weights.pth'))

# Use the loaded model for inference
# ... your inference code here ...
```


**Example 3: Handling Multiple Checkpoints and Model Versioning**

In large projects, managing multiple checkpoints becomes critical.  This example demonstrates a slightly more sophisticated approach utilizing a timestamp to create uniquely named checkpoints.  This greatly simplifies model versioning.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ... (Model and optimizer definition as in Example 1) ...

# ... your training code here ...

timestamp = time.strftime('%Y%m%d_%H%M%S')
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, f'checkpoint_{timestamp}.pth')

# Loading a specific checkpoint would then involve specifying the timestamp in the filename.
```



**3. Resource Recommendations:**

The official PyTorch documentation provides comprehensive and detailed explanations of these functions and their usage.  Further, exploring advanced techniques like using `torch.save` with custom serializers for improved compatibility or utilizing dedicated model versioning tools can greatly benefit large-scale projects.  Lastly, I'd recommend consulting relevant research papers and blog posts discussing best practices for managing model checkpoints in deep learning projects. These additional resources will solidify your understanding and assist in adapting these techniques to your specific needs.  Careful consideration of data structures within the checkpoints and error handling during loading are crucial for robust and reproducible workflows.  Finally, understanding the implications of loading models trained on different hardware architectures can prevent unexpected errors.
