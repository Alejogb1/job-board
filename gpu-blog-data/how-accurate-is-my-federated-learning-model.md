---
title: "How accurate is my Federated Learning model?"
date: "2025-01-26"
id: "how-accurate-is-my-federated-learning-model"
---

Federated Learning (FL) models, by their distributed nature, inherently pose unique challenges when assessing accuracy compared to traditional centralized machine learning. My experience across multiple projects implementing FL for client-side data privacy indicates that accuracy evaluation is not a singular, static value but rather a multi-faceted consideration. Simply reporting a global accuracy score derived from a centralized test set can be misleading. Several factors contribute to the perceived accuracy of an FL model, and a holistic analysis requires looking beyond a single metric.

Firstly, understanding the inherent statistical heterogeneity of federated data is crucial. Unlike centralized datasets where data distributions often conform to a predictable pattern, federated data is typically *non-IID* (non-independent and identically distributed). Each client’s data may represent a distinct sub-population with its own unique characteristics and biases. The model’s accuracy on the global, aggregated dataset (if such a dataset is constructed for testing) may not reflect its performance on individual clients. A model achieving high aggregate accuracy might perform poorly on clients with data significantly different from the aggregate distribution.

Therefore, when assessing the accuracy of an FL model, I first examine *per-client* performance metrics. This approach requires defining suitable metrics based on the task, typically accuracy, precision, recall, F1-score, or area under the ROC curve for classification, or mean absolute error, mean squared error, or R-squared for regression. Instead of a single number, I calculate these metrics for each client individually using the client's hold-out data. This process produces a distribution of performance metrics, highlighting disparities in model accuracy across the federated network.

The analysis usually involves visual exploration via box plots or histograms to display the distribution of accuracy scores, allowing me to assess the spread, median performance, and identify potential outliers or underperforming clients. Secondly, I investigate the client-specific data size. Clients with significantly less data during training may exhibit lower model performance compared to clients with more substantial datasets. This phenomenon is frequently observed across various FL implementations, and it dictates a separate analysis to verify that clients with smaller datasets aren't being penalized by the training procedure.

In addition to per-client performance, it's critical to evaluate the stability of model performance during the training process. Federated learning is an iterative procedure with multiple rounds of local training and aggregation. Plotting the model’s performance (e.g., average accuracy or loss) over these rounds, using a held-out validation dataset, provides crucial information about convergence and the potential for over-fitting. Ideally, the performance on the validation set should plateau within a reasonable number of rounds. A model with unstable or wildly fluctuating performance indicates either problems with the training parameters, hyperparameter tuning, data quality issues, or possibly, the aggregation strategy employed.

Finally, the communication strategy used by the federated algorithm also influences model accuracy. Specifically, techniques like FedAvg, which calculates a simple weighted average of client model updates, can suffer when client data distributions differ significantly. Exploring alternative aggregation strategies, such as those that take into account client-specific model performance, can sometimes lead to improvements in overall accuracy.

Here are three code snippets (Python pseudo-code) to exemplify some of these points:

**Example 1: Per-Client Accuracy Evaluation**

```python
def evaluate_per_client_accuracy(global_model, clients, criterion, metric):
    """
    Evaluates the model's accuracy on each client's local test set.

    Args:
        global_model (Model): The global Federated Learning model.
        clients (list): A list of client objects, each having a test dataset.
        criterion (function): Loss function for model evaluation
        metric (function): Metric to evaluate the model against
    Returns:
        dict: A dictionary of {client_id: metric_value}.
    """
    client_metrics = {}
    for client in clients:
        local_test_loader = client.get_test_data() # get the test dataloader for a given client
        local_model = copy.deepcopy(global_model) # copy global model for evaluation on local test data

        #Evaluate local model on test loader
        local_model.eval()
        total_loss = 0
        total_metric = 0

        with torch.no_grad():
            for inputs, labels in local_test_loader: #assuming input and target labels are available in dataloader
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0) #input size for batch
                total_metric += metric(outputs, labels) * inputs.size(0)

            avg_loss = total_loss / len(local_test_loader.dataset)
            avg_metric = total_metric / len(local_test_loader.dataset)

        client_metrics[client.id] = {"loss": avg_loss, "metric": avg_metric}
    return client_metrics

# Usage (assumes client objects, global model, and metric definition)
client_accuracies = evaluate_per_client_accuracy(global_model, clients, criterion, accuracy)
# Plot the accuracy distribution using matplotlib
# Examine disparities and perform outlier analysis
```

In this function, I implement per-client model evaluation. I iterate through each client, load its local testing data, and evaluate the model's performance against this data. The output is a dictionary of metrics which represents per-client accuracy distribution for analysis. It clearly demonstrates the disparity in performance across the different clients.

**Example 2: Training Stability Analysis**

```python
def train_federated_model(clients, global_model, num_rounds, aggregation_strategy, validation_data_loader):
    """
    Trains the Federated Learning model and tracks the global model performance on the validation set.

    Args:
       clients (list): List of client objects.
       global_model (Model): The global Federated Learning model.
       num_rounds (int): Number of training rounds
       aggregation_strategy (function): Aggregation function.
       validation_data_loader (DataLoader): DataLoader for validation set.

    Returns:
        list: A list containing the validation performance after each training round.
    """
    validation_performance_history = []

    for round_num in range(num_rounds):
        # Local training on each client
        updated_models = []

        for client in clients:
             local_model = copy.deepcopy(global_model) # copy global model for local client updates
             local_model.train()
             local_model = client.train_locally(local_model) # assumes local training implementation
             updated_models.append(local_model)

        # Aggregation of client models
        global_model = aggregation_strategy(updated_models) # use passed in aggregation function

         # Validate the global model
        global_model.eval() # evaluation mode
        total_loss = 0
        total_metric = 0
        criterion = LossFunction # assumed implementation

        with torch.no_grad():
            for inputs, labels in validation_data_loader:
                outputs = global_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_metric += metric(outputs, labels) * inputs.size(0)

        avg_loss = total_loss / len(validation_data_loader.dataset)
        avg_metric = total_metric / len(validation_data_loader.dataset)

        validation_performance_history.append({"round": round_num, "loss": avg_loss, "metric": avg_metric})


    return validation_performance_history

# Usage
# Assuming client and model setup and parameter definition
# Use a validation dataset loader and an aggregation strategy (e.g., FedAvg, FedProx)
# Collect training data from the client class objects.
validation_history = train_federated_model(clients, global_model, num_rounds, aggregation_strategy, validation_data_loader)

#Plot the validation performance over the rounds for stability analysis
#Assess the plot for any fluctuations, improvements, or stability of convergence
```

In the code example above, I’ve focused on showing how to track model validation performance over training rounds. I collect the average performance metrics at each round, storing it in `validation_performance_history`. Using this data, we would visualize a performance versus rounds graph which helps assess the stability and convergence during training of the federated learning model. This step is critical to identify potential problems early in the process.

**Example 3: Data Size Analysis**

```python
def analyze_client_data_size(clients):
    """
    Analyzes the size of each client's training dataset.

    Args:
        clients (list): A list of client objects, each having a training dataset.

    Returns:
        dict: A dictionary of {client_id: dataset_size}.
    """
    client_data_sizes = {}
    for client in clients:
        train_dataset = client.get_train_data() # assuming a function to retrieve the training set
        client_data_sizes[client.id] = len(train_dataset)
    return client_data_sizes

# Usage
client_sizes = analyze_client_data_size(clients)
#Analyze the returned dictionary to determine how skewed the client data distribution is.
#Relate this information with per client accuracy to see if there's a correlation.
```

The above function simply iterates through all the clients to access each client's training set. The function collects the size of each client's training dataset which allows for an analysis of dataset size distribution across the federated network. Comparing the per-client accuracies obtained in example 1 against the data size information from this function, would reveal whether there's a correlation between local dataset size and local performance.

In summary, evaluating the accuracy of an FL model goes beyond a single number. I advocate for a multi-faceted approach that encompasses per-client performance, model stability across rounds, and an analysis of the data distribution among the clients. This holistic strategy provides a more comprehensive picture of the actual performance of the model in a federated environment. For further information, I recommend consulting publications and research material from institutions specializing in distributed computing and machine learning, alongside publications by leading researchers in the FL domain.
