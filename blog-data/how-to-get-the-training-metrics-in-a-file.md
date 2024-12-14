---
title: "How to get the training metrics in a file?"
date: "2024-12-14"
id: "how-to-get-the-training-metrics-in-a-file"
---

alright, so you're asking how to get training metrics dumped into a file, huh? been there, done that, got the t-shirt. or, well, a few crumpled notebooks filled with plots and handwritten notes, more like. it's a common problem when you're knee-deep in training neural networks or any kind of model really.

let me tell you, early on in my career, i thought just printing to the console was enough. i'd have this glorious wall of numbers scrolling by, feeling like a true data scientist wizard, and then… poof, the session ends, the console clears, and i'm left with nothing. so, yeah, i learned the importance of logging to a file, the hard way. the amount of time wasted trying to recreate previous runs because i didn't log the metrics… man, i don’t want to revisit those days.

anyway, let’s break down how i usually handle this now. it mostly boils down to choosing the logging format and library that suits you best, and then adding some hooks into your training loop to save the data. there are a few ways to approach this, so let me give you a few snippets with different options, all of these i actually use and have tweaked to what i need. i'll stick to pretty straightforward python for these examples, since that seems to be the lingua franca for this stuff.

first up, the most basic one: csv. this approach is simple and gives you a table you can easily import into other tools, like pandas, or, dare i say, excel, if you are into it.

```python
import csv

def log_metrics_to_csv(filepath, metrics, header=None):
    """
    logs metrics to a csv file.

    args:
      filepath (str): path to save the csv
      metrics (dict): dictionary of metrics.
        ex: {"epoch":1, "loss":0.12, "accuracy":0.95}
        header (list): header of the file
    """
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header if header else metrics.keys())
        if not file_exists or not header:
            writer.writeheader()
        writer.writerow(metrics)

# example of how to use it
# let’s say within the loop for every epoch you have:
# epoch = 1
# loss = 0.2
# accuracy = 0.8
# and at the end you want a csv to track everything.

metrics = {"epoch":epoch, "loss":loss, "accuracy":accuracy}
log_metrics_to_csv("training_log.csv", metrics, header=["epoch","loss","accuracy"])
```

i have used this so many times that i have a decorator that does this for me when i train any model, i highly recommend you doing that for yourself. this snippet basically appends a new row to the file every time you call it. the `file_exists` check ensures you don't write the header every time you call the function. this is the simplest one that i use for the smaller stuff. very easy to read with pandas too, like so

```python
import pandas as pd

df = pd.read_csv("training_log.csv")
print(df)
```

next up, json. if you are more comfortable with handling json data structures then this is a very solid alternative and something i use pretty frequently if i need to add more complex things to the log than simple metrics.

```python
import json
import os

def log_metrics_to_jsonl(filepath, metrics):
    """
    logs metrics to a jsonl file.

    args:
      filepath (str): path to save the jsonl.
      metrics (dict): dictionary of metrics.
    """
    with open(filepath, 'a') as f:
        json.dump(metrics, f)
        f.write('\n')


# example usage
# within your training loop after each epoch
# you have
# epoch = 1
# loss = 0.2
# accuracy = 0.8
# and you want to track in a jsonl file

metrics = {"epoch":epoch, "loss":loss, "accuracy":accuracy}
log_metrics_to_jsonl("training_log.jsonl", metrics)

```
this writes each set of metrics as a json object to the file, one per line, using jsonl, which is great if you need to read the files later by doing

```python
import json

def read_jsonl_file(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)

for data in read_jsonl_file("training_log.jsonl"):
  print(data)
```
i’ve used this jsonl approach in projects where i wanted to log more than just basic metrics like training time, hyperparameters, input data distribution, and so on. the json format allows for nested data structures, which is great when your logging requirements become more complex.

and finally, if you want something more sophisticated you might consider using tensorboard. while it's not a plain file, it does output files that can be read to visualize the metrics and they are really useful for more complex models. if you're training deep learning models, it's almost a necessity to get good visualizations and insight. it's the only way that my team hasn’t gone mad after a few months.

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

# sample data to simulate a training loop. replace with yours.
inputs = torch.randn(64, 10)
targets = torch.randint(0, 2, (64,))
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
loss_function = nn.CrossEntropyLoss()

# initialize summary writer
writer = SummaryWriter("runs/training_log") # you may want to adjust this depending where your log is

# number of epochs to log
epochs = 10

# train
for epoch in range(epochs):
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = loss_function(outputs, targets)
  loss.backward()
  optimizer.step()
  # logging to tensorboard every epoch
  writer.add_scalar('loss', loss.item(), epoch)
  predictions = torch.argmax(outputs, dim = 1)
  correct = (predictions == targets).sum().item()
  accuracy = correct / targets.shape[0]
  writer.add_scalar('accuracy', accuracy, epoch)
  print(f"epoch:{epoch}, loss:{loss.item()}, accuracy: {accuracy}")


writer.close() # good practice
```

with tensorboard, you need to install the tensorboard package and then run it in the command line after the training, like this `tensorboard --logdir=runs`. it outputs a page with the graphs that is really useful. the nice thing about tensorboard is that it lets you visualize pretty much everything, not just simple numbers. histograms of weights, gradients, and all kinds of stuff are all very helpful to debug why your neural network isn't cooperating, but at this stage we are just creating basic files.

if you are into the theory behind it all, i can recommend checking out the original tensorboard papers, they are very complete. you can search on google scholar for the work related to "tensorboard data visualization".  also, if you want to dive deeper in the techniques of model training, a very solid book is "deep learning" from goodfellow et al, that explains the math of it all and how to debug it.

finally, one tip: be consistent in your logging. choose a format and stick with it across projects (as far as it's appropriate for the projects, of course). also, save not only the metrics themselves, but also the settings you used for that training. that way, if you are like me and you forget the parameters you used to get that nice result in the graph. you can quickly check it, trust me it has happened more often than i want to admit. also, it's not a good idea to log everything in the same file. if you start doing multiple runs it will become messy pretty fast. separate your logging output files per experiment, and use a naming convention that you understand.

there you go, three ways to get those training metrics into a file. i hope this saves you a few hours of lost data and a couple of hair strands. remember: good logging practices are essential for every researcher, or any person that trains machine learning models for that matter. they are like the little breadcrumbs you need to find your way back when things go south with the training. or, like a well-documented codebase, except instead of code, it’s your experiments.
