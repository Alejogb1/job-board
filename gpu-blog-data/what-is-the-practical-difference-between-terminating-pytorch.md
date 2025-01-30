---
title: "What is the practical difference between terminating PyTorch training with Ctrl-C and Ctrl-'\'?"
date: "2025-01-30"
id: "what-is-the-practical-difference-between-terminating-pytorch"
---
The practical difference between terminating a PyTorch training loop with Ctrl-C (SIGINT) and Ctrl-\ (SIGQUIT) stems primarily from how Python handles these signals, and consequently how PyTorch reacts to them. Both signals interrupt the execution of a running program, but their intended uses and default behaviors differ significantly, leading to varying outcomes in the context of a long-running PyTorch training session.

Ctrl-C sends a SIGINT signal, which Python translates into a `KeyboardInterrupt` exception. Typically, a Python program will attempt to handle this exception gracefully, executing `try...except KeyboardInterrupt` blocks if they are defined. This mechanism allows the program to perform cleanup tasks, such as saving the model's state or releasing resources, before shutting down. In the context of PyTorch training, this is essential, as a sudden, unmanaged termination could result in loss of progress, corrupted model weights, or other undesirable states.

On the other hand, Ctrl-\ sends a SIGQUIT signal. This signal is designed for forceful, immediate termination of a process, and generally does not provide an opportunity for cleanup. Python, by default, does not handle the SIGQUIT signal. Instead, the operating system terminates the process immediately, with no chance for Python to execute exception handling or finally blocks. When a PyTorch training loop is terminated via SIGQUIT, therefore, it is equivalent to abruptly killing the process from the operating system perspective. This means the training loop halts at the precise moment the signal is received, without any chance of controlled shutdown. There's no checkpoint save, no cleanup of allocated memory within the GPU; the program ceases to exist.

I recall a specific instance early in my work when I was training a particularly resource-intensive model. Due to a misconfigured early stopping condition, the training was running far longer than anticipated. I initially opted to terminate using Ctrl-C. My training script had a `try...except KeyboardInterrupt` block, which was able to gracefully save the latest checkpoint, even though it had not reached the planned number of epochs. This allowed me to resume my training run later with minimal loss. After experimenting and learning about signal handling, I intentionally terminated a different run with Ctrl-\. The immediate termination, which did not include any resource cleanup, resulted in loss of progress and meant that I had to start the training from scratch again. This incident solidified my understanding of the crucial distinction between the two methods.

To illustrate these differences in code, consider the following snippets:

**Example 1: Handling SIGINT (Ctrl-C)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

try:
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    
    for epoch in range(100):
        inputs = torch.randn(20, 10)
        targets = torch.randn(20, 2)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        time.sleep(0.1) #Simulate lengthy process
        
    print("Training completed successfully!")

except KeyboardInterrupt:
    print("\nTraining interrupted by user (Ctrl-C). Saving checkpoint...")
    torch.save(model.state_dict(), 'interrupted_checkpoint.pth')
    print("Checkpoint saved. Exiting...")

```

In this example, the training loop includes a `try...except KeyboardInterrupt` block. When a user presses Ctrl-C during execution, the script catches this exception, saves the model's state to a file named "interrupted_checkpoint.pth," and then exits gracefully. This prevents data loss from the run and allows us to recover progress.

**Example 2: Effect of SIGQUIT (Ctrl-\)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Starting training...")

for epoch in range(100):
    inputs = torch.randn(20, 10)
    targets = torch.randn(20, 2)
        
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
        
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    time.sleep(0.1) #Simulate lengthy process
        
print("Training completed successfully!") # This line will likely not execute if terminated with Ctrl-\

```
Here, there's no `try...except` block to handle termination. If you execute this code and press Ctrl-\ during training, the program terminates immediately. It won't save any checkpoints, and the final message about "Training completed successfully!" will not be printed. The program simply ends mid-execution at the point in the loop where the interrupt occurred. The Python interpreter's clean-up routines are skipped; PyTorch does not get the chance to clean up after itself.

**Example 3: A more complete scenario with a dedicated signal handler**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import time
import signal
import sys

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
checkpoint_file = 'custom_checkpoint.pth'

def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Saving checkpoint...")
    torch.save(model.state_dict(), checkpoint_file)
    print("Checkpoint saved. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("Starting training...")

try:
    for epoch in range(100):
        inputs = torch.randn(20, 10)
        targets = torch.randn(20, 2)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        time.sleep(0.1)

    print("Training completed successfully!")

except Exception as e:
    print(f"An error occurred during training: {e}")


```

This example provides the same functionality as example 1, but uses a more explicit signal handler via the `signal` module. By setting up a handler for `SIGINT`, the user can press Ctrl-C and the designated function `signal_handler` is invoked, allowing for cleanup routines. Using `signal.signal(signal.SIGINT, signal_handler)` is another valid way to handle signals in python without a `try/except` block. Note that this mechanism does not allow for handling `SIGQUIT` at a python level; that signal always results in an abrupt termination. This reinforces the distinction in signal handling and highlights the flexibility provided by Python's ability to handle `SIGINT` but not the system level `SIGQUIT`.

For anyone working with PyTorch or any long-running Python process, I recommend studying signal handling in Python, particularly the `signal` module and the use of `try...except KeyboardInterrupt` blocks. Standard references on Python programming provide good information on exception handling. For more details on signals at the operating system level, consult system programming textbooks. Further, review documentation on PyTorch and data loading best practices to help prevent data loss. Awareness of these details is critical to creating robust and resilient training pipelines, avoiding the frustration of having to rerun long training processes from scratch.
