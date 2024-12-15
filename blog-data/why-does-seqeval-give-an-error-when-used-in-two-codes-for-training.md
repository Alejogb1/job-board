---
title: "Why does seqeval give an error when used in two codes for training?"
date: "2024-12-15"
id: "why-does-seqeval-give-an-error-when-used-in-two-codes-for-training"
---

alright, so you're running into a seqeval hiccup when using it across two training scripts, huh? yeah, i’ve been there, felt that exact pain. it’s less of a seqeval issue and more about how evaluation metrics, especially the finicky ones like sequence labeling metrics, get handled in distributed or multi-process training scenarios. i've been battling this sort of thing since back when tensorflow was still young. not the 1.0 versions, think pre-eager execution days. i was working on a named entity recognition model, a basic crf, but splitting the training across multiple gpus, and i swear, every other week, my metrics would do the exact same dance of death.

let's unpack this, because there are a couple of potential culprits here. mostly it boils down to data aggregation and state synchronization between your training runs. seqeval, if you’re using the default implementation, assumes a single, isolated environment. when you throw in multiple processes or gpus, you're introducing a bunch of chances for things to get out of whack.

first off, think of seqeval as a collector of predictions and ground truth labels. it needs to see all of those to compute its precision, recall, f1, and so on. it does it line by line with text. now, if you've got two different training scripts, each handling a chunk of your dataset, seqeval in each script will have its own local view of the world. these individual views are completely unaware of each other. they might be evaluating on different parts of your data, or worse, they might not be synchronized well enough, leading to duplicate or missing predictions that they both try to evaluate and you get the errors.

if you are using a library that manages training, you might be tempted to think, that it is taking care of everything. but the truth is most libraries use the simplest method that could work, which can lead to some issues down the line, in special the metrics.

here is a simple example of a function, that evaluates some texts. let's call it `evaluate_text`.

```python
from seqeval.metrics import classification_report, f1_score, accuracy_score
import numpy as np

def evaluate_text(y_true, y_pred):
    """
    Evaluates a set of predicted tags.

    Args:
        y_true: A list of lists of true labels.
        y_pred: A list of lists of predicted labels.

    Returns:
        A dictionary of evaluation metrics.
    """

    report = classification_report(y_true, y_pred, output_dict=True)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        'classification_report': report,
        'f1': f1,
        'accuracy': acc,
    }

if __name__ == '__main__':

    y_true = [
        ['B-PER', 'I-PER', 'O', 'B-LOC'],
        ['B-ORG', 'O', 'B-LOC']
    ]
    y_pred = [
        ['B-PER', 'I-PER', 'O', 'B-LOC'],
        ['B-ORG', 'O', 'I-LOC']
    ]

    results = evaluate_text(y_true, y_pred)
    print(results)
```
this simple python code will evaluate `y_true` with `y_pred` using seqeval. It will work fine because the evaluation is localized, it is done locally, in one single machine.

the root cause is usually how the labels and predictions are gathered before being fed to seqeval. it might involve some steps that are not trivial, specially if you use a framework to train the model. most of them does not use distributed metrics out of the box.

here's where you need to be extra careful: each training script might have its own copy of, say, the `y_true` and `y_pred`. if these aren't synchronized in some way before evaluation, seqeval is just doing its job based on incomplete data. the errors will appear in a format that might not point directly to this issue but a simple look on it will be enough to understand the problem.

i remember one very specific occasion where i had a similar problem. i was working with a transformer based language model and fine tuning it for a sequence tagging task. i thought that by changing the evaluation step i would be ok but no, i had to add a step to ensure that the evaluation step was made only after the data was gathered together. it was a simple oversight that took a whole morning to fix. it was an embarrassing situation.

so, what can you do? several options, depending on your exact setup.

one approach is to use distributed data-parallelism with proper aggregation of evaluation metrics. many libraries like pytorch and tensorflow have mechanisms to help. for example, if using pytorch distributed data parallel, you’d make sure that the evaluation happens only at the main process by passing the correct rank and gathering labels and predictions there. something like this, consider the following example:
```python
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import random

class FakeTextDataset(Dataset):
    """A class that emulates a dataset."""
    def __init__(self, size=100):
       self.size = size
       self.data = []
       for _ in range(size):
           rand_size_text = random.randint(5, 15)
           true_labels = [random.choice(['B-PER', 'I-PER', 'O', 'B-LOC', 'B-ORG']) for _ in range(rand_size_text)]
           pred_labels = [random.choice(['B-PER', 'I-PER', 'O', 'B-LOC', 'B-ORG']) for _ in range(rand_size_text)]
           self.data.append((true_labels, pred_labels))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_distributed(y_true, y_pred, rank, world_size):
    """
    Evaluates predictions in a distributed environment.
    Args:
        y_true: local y_true.
        y_pred: local y_pred.
        rank: the local rank of the process.
        world_size: the total number of processes
    Returns:
        A dictionary of evaluation metrics.
    """

    if rank != 0:
      y_true_global = None
      y_pred_global = None
    else:
      y_true_global = [None for _ in range(world_size)]
      y_pred_global = [None for _ in range(world_size)]


    dist.gather_object(y_true, object_list = y_true_global if rank == 0 else None,  dst = 0)
    dist.gather_object(y_pred, object_list = y_pred_global if rank == 0 else None,  dst = 0)

    if rank != 0:
       return None

    y_true_global = [item for sublist in y_true_global for item in sublist if item is not None]
    y_pred_global = [item for sublist in y_pred_global for item in sublist if item is not None]


    report = classification_report(y_true_global, y_pred_global, output_dict=True)
    f1 = f1_score(y_true_global, y_pred_global)
    acc = accuracy_score(y_true_global, y_pred_global)
    return {
        'classification_report': report,
        'f1': f1,
        'accuracy': acc,
    }



if __name__ == '__main__':
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a fake dataset
    dataset = FakeTextDataset(size=100)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Collect predictions and true labels locally
    y_true_local = []
    y_pred_local = []

    for true_labels, pred_labels in dataloader:
        y_true_local.extend(true_labels)
        y_pred_local.extend(pred_labels)

    # Evaluate on the main rank
    results = evaluate_distributed(y_true_local, y_pred_local, rank, world_size)
    if rank == 0:
        print(results)

    dist.destroy_process_group()
```

here, `gather_object` method will receive each `y_true` and `y_pred` in each process, in the main process will group all of them and evaluate on them. the evaluation happens in the main process and the metrics are correct. if you are using something like tensorflow, there are similar ways to gather metrics, using distributed strategy.

another strategy is to use an external metric aggregation service, and i've found this is very useful when debugging this sort of stuff. you send your metrics to an external service (like tensorboard, or mlflow) and the service is the one that does the aggregation and the plotting.

in general, the problem with the evaluation and the error you are facing is because seqeval is looking at a partial view of the training process, which is expected because each one is training a part of the dataset.

and if you have a model, that has multiple outputs, you can aggregate the output metrics in the same way i showed previously.
```python
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from seqeval.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import random

class FakeTextDataset(Dataset):
    """A class that emulates a dataset."""
    def __init__(self, size=100):
       self.size = size
       self.data = []
       for _ in range(size):
           rand_size_text = random.randint(5, 15)
           true_labels1 = [random.choice(['B-PER', 'I-PER', 'O', 'B-LOC', 'B-ORG']) for _ in range(rand_size_text)]
           pred_labels1 = [random.choice(['B-PER', 'I-PER', 'O', 'B-LOC', 'B-ORG']) for _ in range(rand_size_text)]
           true_labels2 = [random.choice(['B-LOC', 'I-LOC', 'O', 'B-ORG', 'B-MISC']) for _ in range(rand_size_text)]
           pred_labels2 = [random.choice(['B-LOC', 'I-LOC', 'O', 'B-ORG', 'B-MISC']) for _ in range(rand_size_text)]

           self.data.append(((true_labels1,true_labels2), (pred_labels1,pred_labels2)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_distributed_multiple_outputs(y_true_all, y_pred_all, rank, world_size):
    """
    Evaluates predictions in a distributed environment, for a model with multiple outputs.
    Args:
        y_true_all: local y_true, a list of tuples.
        y_pred_all: local y_pred, a list of tuples.
        rank: the local rank of the process.
        world_size: the total number of processes
    Returns:
        A dictionary of evaluation metrics.
    """

    if rank != 0:
      y_true_global = None
      y_pred_global = None
    else:
      y_true_global = [None for _ in range(world_size)]
      y_pred_global = [None for _ in range(world_size)]


    dist.gather_object(y_true_all, object_list = y_true_global if rank == 0 else None,  dst = 0)
    dist.gather_object(y_pred_all, object_list = y_pred_global if rank == 0 else None,  dst = 0)

    if rank != 0:
       return None

    y_true_global = [item for sublist in y_true_global for item in sublist if item is not None]
    y_pred_global = [item for sublist in y_pred_global for item in sublist if item is not None]

    all_results = {}

    for i in range(len(y_true_global[0])):
        y_true_output = [output[i] for output in y_true_global]
        y_pred_output = [output[i] for output in y_pred_global]

        report = classification_report(y_true_output, y_pred_output, output_dict=True)
        f1 = f1_score(y_true_output, y_pred_output)
        acc = accuracy_score(y_true_output, y_pred_output)

        all_results[f'output_{i}'] = {
            'classification_report': report,
            'f1': f1,
            'accuracy': acc,
        }

    return all_results



if __name__ == '__main__':
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create a fake dataset
    dataset = FakeTextDataset(size=100)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Collect predictions and true labels locally
    y_true_local = []
    y_pred_local = []

    for (true_labels1, true_labels2), (pred_labels1, pred_labels2) in dataloader:
        y_true_local.extend(list(zip(true_labels1, true_labels2)))
        y_pred_local.extend(list(zip(pred_labels1, pred_labels2)))


    # Evaluate on the main rank
    results = evaluate_distributed_multiple_outputs(y_true_local, y_pred_local, rank, world_size)
    if rank == 0:
        print(results)

    dist.destroy_process_group()
```
this example shows how to treat multiple outputs of the model, it has a few extra steps, but it serves to illustrate the need to adapt the evaluation process to distributed training. i've seen models with upwards of 5 different outputs, each one needs to be individually aggregated and evaluated. i still remember a particular case where the evaluation was failing silently because the multiple outputs were not handled, and i wasted a whole week trying to debug it. it is funny now, but wasn't at the time.

for resources, besides the documentation of your framework, i strongly recommend looking into "deep learning with pytorch" by elias and others. it has a very clear explanation of distributed training and how to deal with metrics, and "distributed training of deep neural networks" by ben-nun and others. this paper is a classic, but the underlying theory it explains is very applicable to your problem. it really is worth the time to read these materials.

the takeaway is that seqeval isn’t the issue. it's just doing what it is meant to do on what it is given. the real issue is the distributed nature of the training and you need to collect your predictions and labels correctly in the correct order, so seqeval can do its job. it is usually a data aggregation and synchronization issue. it is a common mistake, it happened to me, it probably happened to every other experienced machine learning engineer. hopefully, you can solve your problem fast.
