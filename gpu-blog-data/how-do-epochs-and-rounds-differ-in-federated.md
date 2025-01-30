---
title: "How do epochs and rounds differ in federated learning?"
date: "2025-01-30"
id: "how-do-epochs-and-rounds-differ-in-federated"
---
Federated learning, at its core, is a distributed machine learning paradigm; however, understanding the nuanced interplay between epochs and rounds is paramount for its effective implementation. They are not interchangeable concepts, and their distinct roles govern how models learn in a decentralized manner. Confusion often arises because both involve iterative processing, but they operate at different scales within the federated setting.

An **epoch**, within the local training context of a participating client device, signifies one complete pass through the client’s entire local dataset during a single training iteration. This is analogous to traditional, centralized machine learning, where an epoch represents the model learning from all available training examples. The key distinction in federated learning is that each client performs this local training *independently*, with its own data, during its turn in the federated learning cycle.

In contrast, a **round**, often also referred to as a communication round, represents one complete cycle of the federated learning process. This cycle encompasses the server initializing a model, distributing it to participating clients, facilitating their local training (which involves a predefined number of epochs by each client), aggregating the updated model parameters or gradients, and finally updating the server’s global model. The round is about server-client interaction and the collaborative learning process, not just a single client’s data pass. Think of it as the high-level synchronization mechanism for all participating clients.

Thus, several epochs typically happen *within* a single federated learning round. The number of local epochs is a hyperparameter controlled by the federated learning system designer, affecting the balance between local learning and global model convergence. More local epochs can lead to clients overfitting to their local, potentially biased data, hindering the global model's generalization. Conversely, too few local epochs may result in insufficient local learning, slowing overall convergence.

To illustrate, consider the federated averaging (FedAvg) algorithm. Below are code snippets focusing on the client and server operations, demonstrating the relationship between epochs and rounds. These are simplified python examples using pseudo-code concepts for illustration, not functional implementations:

**Example 1: Client-side Local Training Loop**

```python
def client_local_training(model, local_dataset, learning_rate, local_epochs):
    """
    Performs local training on a client's dataset for a given number of epochs.

    Args:
        model: The current global model.
        local_dataset: The client's local dataset.
        learning_rate: Learning rate for the client's optimizer.
        local_epochs: The number of local epochs to train for.

    Returns:
        The updated model weights after local training.
    """
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = LossFunction()  # Placeholder for the actual loss
    
    for epoch in range(local_epochs):
        for batch in local_dataset: # Iterating through local data
            optimizer.zero_grad()
            predictions = model(batch.features)
            loss = criterion(predictions, batch.labels)
            loss.backward()
            optimizer.step()
    
    return model.state_dict() # Return weights, not the whole model
```
This client-side code demonstrates how an individual client trains its local copy of the global model for a specified number of epochs using its local dataset. The inner loop iterating over the local dataset represents the core functionality of an epoch. Notice that multiple of these inner loops occur during a single invocation of  `client_local_training`. The function return is the model's state dictionary. This avoids sending the whole model object and is more efficient. This updated state dictionary is sent back to the server after local training.

**Example 2: Server-side Federated Aggregation within a Round**

```python
def federated_averaging(global_model, clients, aggregation_weights):
    """
    Performs federated averaging over the client models.

    Args:
      global_model: The current global model.
      clients: A list of client objects, each representing one participating device.
      aggregation_weights: Weights for each client based on size of local data

    Returns:
      The updated global model state.
    """
    
    client_updates = [] # Place to hold the weight updates from each client
    for client,weight in zip(clients,aggregation_weights):
       client_updates.append((client.get_local_update(),weight)) # each update is tied to the client's aggregation weight

    aggregated_state = aggregate_client_updates(client_updates) # Using custom function to perform the weighted average
    global_model.load_state_dict(aggregated_state)
    return global_model.state_dict()

def aggregate_client_updates(client_updates):
    """ 
        Aggregates weight updates from clients with their associated aggregation weights
    """
    
    aggregated_weights = {}
    for update, weight in client_updates:
        for key, value in update.items(): # Iterate through parameter weights
            if key not in aggregated_weights:
                 aggregated_weights[key] = torch.zeros_like(value)
            aggregated_weights[key] += weight*value # Weighted sum
            
    return aggregated_weights
```
This snippet illustrates the server's role. It collects local model updates from clients after they have each trained for their defined number of epochs. The function `federated_averaging` orchestrates this aggregation process. It is called once per round. The  `aggregate_client_updates` function combines all the client updates, weighted by the amount of local training data.  This is a critical step in federated learning to generate the new global model weights. This update is then used to update the server model state. Note that this entire function call occurs within one federated learning round.

**Example 3: The High Level Federated Learning Loop**

```python

def federated_learning_loop(global_model, clients, learning_rate, local_epochs, num_rounds):
    """
    Performs federated learning over a given number of rounds.

    Args:
      global_model: The global model.
      clients: A list of client objects, each with a local training method.
      learning_rate: Learning rate for client's local training.
      local_epochs: The number of epochs for each local client.
      num_rounds: Total number of training rounds for the system.
    """
    for round in range(num_rounds):
        print(f"Starting Round {round+1}")
        # Server disseminates model
        for client in clients:
            client.set_global_model(global_model)
        
        # Clients perform local training
        for client in clients:
            client.local_train(learning_rate, local_epochs)
            
        # Server aggregates results
        aggregation_weights = [len(client.dataset) for client in clients] # Weight by the size of local dataset
        aggregated_model_state = federated_averaging(global_model, clients, aggregation_weights)
        global_model.load_state_dict(aggregated_model_state) # Update server model
        print(f"Round {round+1} complete.")
    
    return global_model

```

This is a higher-level view of the overall federated learning process.  It directly shows that the training process is iterated over rounds. The key takeaway here is that for each round, a number of *local epochs* occurs on each client *prior* to server aggregation. Each iteration of the outer for loop corresponds to a single round of federated learning. The inner for loops, such as for updating the models, happen once per round. This explicitly distinguishes the high-level system-wide round from the local client's internal epoch.

In summary, epochs are local training iterations within each client's device.  Rounds, conversely, are global iterations of the entire federated learning process, involving communication and aggregation. The number of epochs per round influences the convergence speed and the generalization performance of the model. These parameters need careful tuning for optimal performance, depending on the use case, heterogeneity of client devices, and datasets. Understanding this distinction is crucial for effective system design and debugging. I have observed many instances where incorrect training behaviors result from improperly configuring the number of rounds and local epochs.

For further information, resources that provide a conceptual understanding of federated learning architecture, algorithms like FedAvg, and practical implementation guidelines are highly recommended. Academic publications focusing on distributed machine learning, particularly in the context of privacy-preserving techniques, offer rigorous mathematical formulations and deeper insights. Framework documentation for popular federated learning libraries, such as those from Google and PyTorch, is also beneficial for real-world application.  Open-source repositories providing practical code examples can offer implementation level understanding and context.
