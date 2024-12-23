---
title: "How can client data be partitioned into training, testing, and client sets using TFF?"
date: "2024-12-23"
id: "how-can-client-data-be-partitioned-into-training-testing-and-client-sets-using-tff"
---

Okay, let’s talk about partitioning data for federated learning with TensorFlow Federated (tff). I've spent a fair bit of time grappling with this, especially back when I was working on that distributed anomaly detection project for the retail sector. We had data coming in from hundreds of stores, each with its own unique patterns and limitations. So, proper partitioning was absolutely critical for building a robust and generalizable model. Getting it wrong meant a model that would be essentially worthless when deployed.

When you're dealing with federated learning, the idea of partitioning data isn’t just about train/test/validation splits in the traditional sense. It's much more granular and involves another dimension: the clients themselves. We need to consider that our 'data' is distributed across multiple clients, each holding a subset of the overall dataset, and that we need to partition this data both within and across these client boundaries for effective model training and evaluation. In practice, this means thinking about how you'll use data from each client, some for training, some perhaps for local testing, and some, in some advanced scenarios, for evaluating the global model.

The basic concept is this: we need to create three logical sets of data: *training data*, which is used to fit the model; *testing data*, which is used to evaluate the performance of the model after training is done; and *client data*, which are the individual client datasets distributed across the population of participating devices. In traditional machine learning, partitioning this data is pretty straightforward. In federated learning, we have to handle a situation where each client may not have all the categories of data, and our splits need to reflect that. Further, the goal often isn't just to train on a subset and test on another. Sometimes, we want to have local test data to let individual clients test their locally trained model.

Let’s break it down with some code examples to clarify how we might implement this using TFF. Imagine we have a dataset of images, representing a simplified version of what I’ve seen in my experience.

**Example 1: Basic Client-Specific Train/Test Split**

This first example shows the common scenario where, for each client, you divide their local dataset into a local training and local testing set:

```python
import tensorflow as tf
import tensorflow_federated as tff

def create_client_datasets(client_ids, dataset, train_ratio=0.8):
    client_datasets = []
    for client_id in client_ids:
        client_data = dataset.create_tf_dataset_for_client(client_id)
        client_size = len(list(client_data))
        train_size = int(client_size * train_ratio)
        
        train_data = client_data.take(train_size)
        test_data = client_data.skip(train_size)

        client_datasets.append((train_data, test_data))
    return client_datasets

def preprocess(element):
    return element['image'], element['label']


def create_federated_datasets(client_datasets):
   federated_train_data = [train.map(preprocess).batch(32) for train, _ in client_datasets]
   federated_test_data = [test.map(preprocess).batch(32) for _, test in client_datasets]
   return federated_train_data, federated_test_data

# Mock dataset (replace with your actual data)
example_dataset = tff.simulation.datasets.emnist.load_data()

client_ids = example_dataset.client_ids[0:3] # using only a small number of clients for the demo
client_datasets = create_client_datasets(client_ids, example_dataset)
federated_train_data, federated_test_data = create_federated_datasets(client_datasets)

print(f"First client's training set contains {len(list(federated_train_data[0]))} batches.")
print(f"First client's test set contains {len(list(federated_test_data[0]))} batches.")
```

In this snippet, `create_client_datasets` splits the data for each client into local training and testing sets based on a `train_ratio`. We iterate through all the clients, perform the split using `.take()` and `.skip()`, and return a list of tuples representing the train and test datasets of each client. The `create_federated_datasets` function then transforms these local tf.data.Dataset objects into a list of batchable datasets that are usable for federated learning. The `preprocess` function allows us to select the image and labels that we will use for the training process. This represents the most common split: each client has their own local split.

**Example 2: Using a Hold-Out Client Group for Global Testing**

This next example adds complexity by creating a separate set of client data (hold-out clients) that are never used for training, and exclusively used for global model evaluation:

```python
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

def create_client_datasets_with_holdout(client_ids, dataset, train_ratio=0.8, holdout_ratio = 0.2):
    np.random.shuffle(client_ids)
    holdout_size = int(len(client_ids) * holdout_ratio)
    holdout_clients = client_ids[0:holdout_size]
    train_clients = client_ids[holdout_size:]
    
    client_train_datasets = []
    client_test_datasets = []
    client_holdout_datasets = []
    
    for client_id in train_clients:
        client_data = dataset.create_tf_dataset_for_client(client_id)
        client_size = len(list(client_data))
        train_size = int(client_size * train_ratio)
        
        train_data = client_data.take(train_size)
        test_data = client_data.skip(train_size)

        client_train_datasets.append(train_data)
        client_test_datasets.append(test_data)
        
    for client_id in holdout_clients:
        holdout_data = dataset.create_tf_dataset_for_client(client_id)
        client_holdout_datasets.append(holdout_data)

    return client_train_datasets, client_test_datasets, client_holdout_datasets
    
def preprocess(element):
    return element['image'], element['label']


def create_federated_datasets(client_train_datasets, client_test_datasets, client_holdout_datasets):
   federated_train_data = [train.map(preprocess).batch(32) for train in client_train_datasets]
   federated_test_data = [test.map(preprocess).batch(32) for test in client_test_datasets]
   federated_holdout_data = [holdout.map(preprocess).batch(32) for holdout in client_holdout_datasets]
   return federated_train_data, federated_test_data, federated_holdout_data


example_dataset = tff.simulation.datasets.emnist.load_data()

client_ids = example_dataset.client_ids[0:10] # using only a small number of clients for the demo
client_train_data, client_test_data, client_holdout_data = create_client_datasets_with_holdout(client_ids, example_dataset)
federated_train_data, federated_test_data, federated_holdout_data = create_federated_datasets(client_train_data, client_test_data, client_holdout_data)


print(f"Number of clients for training is {len(federated_train_data)}.")
print(f"Number of clients for holdout is {len(federated_holdout_data)}.")

```

Here, we introduce `create_client_datasets_with_holdout`. A portion of clients is selected randomly to be a part of the "holdout" group, whose data will not participate in any stage of training, but will only be used to test the globally trained model. This approach gives a much better idea of how the model would generalize to a previously unseen group of clients. We then use `create_federated_datasets` to produce datasets suitable for federated learning. This more advanced scenario helps to evaluate how well the model generalizes beyond the original set of clients.

**Example 3: Stratified Sampling within Clients**

A more advanced strategy might require stratified sampling within each client's dataset. Imagine that the clients have different data distributions and that you want to guarantee that each client contributes samples of each label to the overall training dataset. This approach can be critical for ensuring that the model is robust to variations in client data distribution:

```python
import tensorflow as tf
import tensorflow_federated as tff
import collections

def create_stratified_client_datasets(client_ids, dataset, samples_per_label=5, labels = [0,1,2,3,4,5,6,7,8,9]):

    client_datasets = []
    for client_id in client_ids:
        client_data = dataset.create_tf_dataset_for_client(client_id)
        
        stratified_dataset = []
        for label in labels:
            label_data = client_data.filter(lambda x: x['label'] == label).take(samples_per_label)
            stratified_dataset.append(label_data)
            
        combined_dataset = tf.data.Dataset.from_tensor_slices(tf.concat([list(x) for x in stratified_dataset],0))
        client_datasets.append(combined_dataset)
    return client_datasets
    
def preprocess(element):
    return element['image'], element['label']
    
def create_federated_datasets(client_datasets):
    federated_train_data = [train.map(preprocess).batch(32) for train in client_datasets]
    return federated_train_data

example_dataset = tff.simulation.datasets.emnist.load_data()

client_ids = example_dataset.client_ids[0:3]
client_datasets = create_stratified_client_datasets(client_ids, example_dataset)
federated_train_data = create_federated_datasets(client_datasets)

print(f"First client's dataset contains {len(list(federated_train_data[0]))} batches.")
```
Here, `create_stratified_client_datasets` is introduced. This function creates a dataset where each client contributes a fixed number of examples per label.  We iterate through labels, filter data that belongs to each label, and take `samples_per_label` from each one. Then we concatenate all the examples into a new dataset. This method can help in situations where some labels are not uniformly distributed across clients.

**Further reading**

I’d highly recommend delving into the TFF documentation, especially the tutorials on data preprocessing. Specifically, the *TensorFlow Federated API Documentation* is the primary source of truth. You can also delve into some books on federated learning. The book *Federated Learning* by Yang et al. provides a very detailed explanation of different partitioning strategies and common practical challenges. Further, publications from the Federated Learning Research group from Google (which is where tff originated) are also essential reads.

In summary, partitioning data in TFF for federated learning is a multifaceted process, and the way you structure it directly impacts how well your models will perform. You need to account for the client's local data distributions, decide if you want hold-out clients for testing, and consider the benefits of stratified sampling if data is unbalanced. It's an iterative process, and the best approach will depend entirely on the peculiarities of the task at hand. Hope this helps.
