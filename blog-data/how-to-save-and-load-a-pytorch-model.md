---
title: "How to save and load a PyTorch model?"
date: "2024-12-16"
id: "how-to-save-and-load-a-pytorch-model"
---

Let’s tackle this, shall we? It's a question that might seem straightforward on the surface but quickly reveals subtleties as you delve deeper into practical deployment scenarios. I’ve spent a fair amount of time optimizing model workflows, and dealing with the nuances of saving and loading PyTorch models efficiently is something I've had to refine over many projects. It’s not just about making things work, it's about making them robust, repeatable, and maintainable.

The core challenge, as I see it, is that a PyTorch model isn't just a set of weights. It’s an intricately constructed computational graph, and we need to preserve both its architecture and learned parameters for later reuse. There are essentially two primary approaches you'll encounter in most projects: saving and loading the entire model directly, or saving and loading only the state dictionary. Each has its place, advantages, and disadvantages.

The first approach, saving the entire model instance, involves serializing the model object using `torch.save`. This is usually the simplest method initially. It captures everything about the model, its structure, and the trained parameters. However, the key issue here is the reliance on the availability of the model’s class definition during loading. You must have the exact same class code at the time of loading as you did at the time of saving. It's a good starting point for basic tasks or when you control both training and loading environments. Here's a quick demonstration:

```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize model and training (dummy)
model = SimpleNet(10, 5, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dummy_input = torch.randn(1, 10)
dummy_output = model(dummy_input)
loss_fn = nn.MSELoss()
loss = loss_fn(dummy_output, torch.randn(1, 2))
loss.backward()
optimizer.step()

# Save the entire model
torch.save(model, 'entire_model.pth')

# Load the entire model
loaded_model = torch.load('entire_model.pth')

# Check if loaded model produces the same results
with torch.no_grad():
    loaded_output = loaded_model(dummy_input)

assert torch.all(loaded_output == dummy_output)
print('Entire Model Saved and Loaded Correctly.')
```

In the above code, we’ve defined `SimpleNet`, created an instance, performed some dummy training and then used `torch.save` to persist it as 'entire_model.pth'. Subsequently, we used `torch.load` to retrieve the model and checked to confirm its functionality. Now, this works fine in simple use cases, but in practical projects where you might be refactoring code or deploying models on different servers, version conflicts can occur if your class definition doesn’t perfectly align during loading.

The alternative, and often the preferred approach in more robust and scalable systems, is to save only the model’s state dictionary. This dictionary, accessed by `model.state_dict()`, contains just the trainable parameters for each layer. Critically, this approach decouples model architecture from model weights. This gives you a huge amount of flexibility because you can initialize a model instance, load in the weights, and the model can perform its task. Think about it like a set of instructions that can be followed by different tools as long as the tools have the right structure.

Here's how it's done in practice:

```python
import torch
import torch.nn as nn

# Re-using the same SimpleNet class

model_state = SimpleNet(10, 5, 2)
optimizer_state = torch.optim.Adam(model_state.parameters(), lr=0.001)
dummy_input = torch.randn(1, 10)
dummy_output = model_state(dummy_input)
loss_fn = nn.MSELoss()
loss = loss_fn(dummy_output, torch.randn(1, 2))
loss.backward()
optimizer_state.step()

# Save only the state dictionary
torch.save(model_state.state_dict(), 'model_state.pth')

# Initialize a new model of the same type, then load weights
loaded_model_state = SimpleNet(10, 5, 2)
loaded_model_state.load_state_dict(torch.load('model_state.pth'))
loaded_model_state.eval()  # Set to eval mode for inference

# Check if loaded model produces the same results
with torch.no_grad():
    loaded_output_state = loaded_model_state(dummy_input)

assert torch.all(loaded_output_state == dummy_output)
print('State Dictionary Saved and Loaded Correctly.')
```

Notice how we instantiated the `SimpleNet` again before using `load_state_dict`. We load the state dictionary using `loaded_model_state.load_state_dict` and then switch the loaded model to evaluation mode using `.eval()` because we are not going to be training and we do not want to change the state of the batch norm, dropout, etc. This method is incredibly valuable, as it means you can, for example, load weights trained on one machine using a slightly different (yet structurally compatible) version of a model on another machine, especially after refactoring your code.

The real-world complexity often comes from wanting to save more than just the model’s weights. Typically, you’ll also want to persist optimizer state, learning rate schedules, or other relevant information for restarting training from a previous checkpoint. In such cases, I often bundle this all into a single dictionary before saving. That's what gives you the power to resume long training runs or to keep track of experiments. Here's how I often handle it:

```python
import torch
import torch.nn as nn

# Re-using the same SimpleNet class

model_bundle = SimpleNet(10, 5, 2)
optimizer_bundle = torch.optim.Adam(model_bundle.parameters(), lr=0.001)
dummy_input = torch.randn(1, 10)
dummy_output = model_bundle(dummy_input)
loss_fn = nn.MSELoss()
loss = loss_fn(dummy_output, torch.randn(1, 2))
loss.backward()
optimizer_bundle.step()

checkpoint = {
    'model_state_dict': model_bundle.state_dict(),
    'optimizer_state_dict': optimizer_bundle.state_dict(),
    'epoch': 50,  # Example of additional info to track
    'loss': loss.item() # Example of tracking a loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Load everything from the checkpoint file
loaded_checkpoint = torch.load('checkpoint.pth')

loaded_model_bundle = SimpleNet(10, 5, 2)
loaded_optimizer_bundle = torch.optim.Adam(loaded_model_bundle.parameters(), lr=0.001)

loaded_model_bundle.load_state_dict(loaded_checkpoint['model_state_dict'])
loaded_optimizer_bundle.load_state_dict(loaded_checkpoint['optimizer_state_dict'])

# Verify we've restored model by evaluating
loaded_model_bundle.eval()
with torch.no_grad():
    loaded_output_bundle = loaded_model_bundle(dummy_input)

assert torch.all(loaded_output_bundle == dummy_output)
print("Checkpoint Saved and Loaded Correctly")
```
With this, not only are we saving model weights, but also optimizer state and other relevant details that are useful for continuation. When you reload this dictionary, everything that was previously in there is available.

For further deep diving, I recommend exploring the PyTorch documentation very thoroughly. Additionally, reading "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann is beneficial for a holistic understanding of best practices in PyTorch development. Also, the "Dive into Deep Learning" book by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola provides excellent conceptual grounding on deep learning which is very helpful for making these save and load workflows smooth and effective. Finally, check the documentation related to `torch.save`, `torch.load`, and `state_dict` on the PyTorch official website for the most updated usage details. These resources combined will give you a really solid grasp of how to handle model persistence effectively in a range of real-world applications.
