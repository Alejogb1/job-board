---
title: "How can pre-trained models be combined into voting ensembles using Python?"
date: "2025-01-30"
id: "how-can-pre-trained-models-be-combined-into-voting"
---
The efficacy of combining pre-trained models through voting ensembles stems from the principle that different models, trained on diverse data or employing varied architectures, will likely capture distinct aspects of the underlying data distribution. This divergence leads to a reduction in overall prediction error when their individual outputs are aggregated. My experience building predictive maintenance systems has shown me that a well-constructed ensemble often outperforms any single constituent model, provided their weaknesses are not perfectly correlated.

A voting ensemble operates by collecting the predictions of multiple trained models for a given input and then aggregating those predictions using a predefined rule to arrive at a final prediction. The aggregation method is crucial, and the two primary types are hard voting and soft voting. Hard voting, the simpler of the two, assigns the final label based on the majority class predicted by the contributing models. Soft voting, on the other hand, averages the predicted probabilities of each class across all models, and the final label corresponds to the class with the highest average probability. Soft voting generally yields better performance when the models are reasonably calibrated in their probability estimates, as it accounts for the confidence levels of each prediction, rather than simply their class labels.

Implementation in Python, specifically using libraries like `scikit-learn` and potentially `torch` or `tensorflow` for the underlying pre-trained models, is relatively straightforward. The process involves several steps. First, load your pre-trained models. Second, create a function that takes an input and passes it through all models, collecting their predictions. Third, implement either hard or soft voting on these predictions. Finally, evaluate the ensemble performance on a held-out dataset.

Let's consider a scenario where we have three pre-trained models for a text classification problem. Assume for simplification that our models, after loading, are encapsulated into predict functions adhering to a common signature. This simplifies the ensemble construction.

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Mock predict functions representing our pre-trained models
def model1_predict(text_input):
    # Assume model1's output are class labels (0 or 1 in this example)
    # Simulating some classification
    return np.random.randint(0, 2)

def model2_predict(text_input):
     # Assume model2's output are class labels (0 or 1 in this example)
    return np.random.randint(0, 2)

def model3_predict(text_input):
    # Assume model3's output are class labels (0 or 1 in this example)
    return np.random.randint(0, 2)

def hard_voting_ensemble(text_input, models):
    predictions = [model(text_input) for model in models]
    # Using bincount for majority voting
    counts = np.bincount(predictions)
    return np.argmax(counts)

# Example usage
models = [model1_predict, model2_predict, model3_predict]
text_sample = "This is an example text"
ensemble_prediction = hard_voting_ensemble(text_sample, models)
print(f"Hard voting ensemble prediction: {ensemble_prediction}")


# Example evaluation with mocked actual labels
actual_labels = np.random.randint(0,2,100) #Mock actual label for evalutaion
ensemble_predictions = [hard_voting_ensemble(text_sample, models) for _ in range(100)]
accuracy = accuracy_score(actual_labels, ensemble_predictions)
print(f"Hard voting ensemble accuracy: {accuracy:.4f}")
```

This example illustrates a basic hard voting implementation. The `hard_voting_ensemble` function accepts a text input and a list of prediction functions (our models), obtains predictions from each, uses `bincount` to determine the frequency of each class prediction and returns the class with the highest frequency. The example also includes some basic evaluation showing the usage of accuracy scoring. The mock labels and predictions are used to demonstrate the process, and should not be considered reflective of real-world model performance. The function `hard_voting_ensemble` is straightforward and easy to adapt when working with classification tasks when the underlying models return class labels.

Next, consider soft voting. Soft voting requires models to return probabilities for each class. Let's modify the mock models to produce probability vectors and adapt the ensemble accordingly.

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Mock predict functions for probability outputs
def model1_predict_soft(text_input):
    # Return probablities for 2 classes
    return np.random.rand(2) #Probability vector

def model2_predict_soft(text_input):
    return np.random.rand(2)

def model3_predict_soft(text_input):
     return np.random.rand(2)

def soft_voting_ensemble(text_input, models):
     # Collecting probabilities from each model
    probabilities = [model(text_input) for model in models]
    # Average probabilities across models
    averaged_probabilities = np.mean(probabilities, axis=0)
    # Return the class with highest average probability
    return np.argmax(averaged_probabilities)

# Example usage of soft voting ensemble
models_soft = [model1_predict_soft, model2_predict_soft, model3_predict_soft]
text_sample = "This is another example"
ensemble_prediction_soft = soft_voting_ensemble(text_sample, models_soft)
print(f"Soft voting ensemble prediction: {ensemble_prediction_soft}")


# Example evaluation with mocked actual labels
actual_labels_soft = np.random.randint(0,2,100) #Mock actual label for evaluation
ensemble_predictions_soft = [soft_voting_ensemble(text_sample, models_soft) for _ in range(100)]
accuracy_soft = accuracy_score(actual_labels_soft, ensemble_predictions_soft)
print(f"Soft voting ensemble accuracy: {accuracy_soft:.4f}")
```

In this soft voting implementation, each model returns a vector of class probabilities using `np.random.rand(2)`. The `soft_voting_ensemble` function collects these probability vectors, calculates their average, and then returns the class index corresponding to the highest average probability.  Again, the provided mock example provides a basic functionality and is not representative of real-world model performance. This approach usually offers improved accuracy compared to hard voting, particularly when using models with varying confidence levels.

Finally, it’s crucial to address how to incorporate pre-trained models from frameworks like `torch` or `tensorflow`. While the overall voting ensemble logic remains the same, the `predict` function now needs to encapsulate the model loading and prediction mechanisms from these frameworks. This can be done by encapsulating the framework-specific operations in the custom prediction function. Let’s use `torch` for demonstration purposes, assuming we have a pre-trained text classification model using the Transformers library.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score

# Mock model class from Transformer library. Should be replaced with your actual model loading
class MockModel():
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels = 2)
        self.device = "cpu"
    def predict(self,text_input):
            self.model.to(self.device)
            self.model.eval()

            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Probability outputs
                probabilities = torch.softmax(outputs.logits, dim=1).cpu().detach().numpy()

            return probabilities[0] #Returns probabilities for class 1 and class 2

# Creating the models
model_torch_1 = MockModel()
model_torch_2 = MockModel()
model_torch_3 = MockModel()

def soft_voting_ensemble_torch(text_input, models):
    probabilities = [model.predict(text_input) for model in models]
    averaged_probabilities = np.mean(probabilities, axis=0)
    return np.argmax(averaged_probabilities)

#Example usage
models_torch = [model_torch_1, model_torch_2, model_torch_3]
example_text_torch = "This is a text classification task example"
ensemble_prediction_torch = soft_voting_ensemble_torch(example_text_torch, models_torch)
print(f"Soft voting ensemble prediction with torch model : {ensemble_prediction_torch}")

# Example evaluation with mocked actual labels
actual_labels_torch = np.random.randint(0,2,100) #Mock actual label for evaluation
ensemble_predictions_torch = [soft_voting_ensemble_torch(example_text_torch, models_torch) for _ in range(100)]
accuracy_torch = accuracy_score(actual_labels_torch, ensemble_predictions_torch)
print(f"Soft voting ensemble accuracy with torch model: {accuracy_torch:.4f}")
```

Here, the `MockModel` demonstrates the integration with PyTorch's Transformers. This `MockModel` class simulates loading and predicting with a pre-trained transformer, handling the tokenization and device placement. The `soft_voting_ensemble_torch` then calls the framework-specific predict function of the models and computes predictions based on these probabilities, as before. Note that for a real implementation, model loading and the prediction mechanism should not be encapsulated using a mocked class. The mock class is for demonstration purposes and not a functional implementation. The primary takeaway is that these details are contained within the specific model classes.

For further study, the documentation for `scikit-learn` provides extensive information on model evaluation metrics. For a deeper understanding of transformer based models, resources from Hugging Face are highly recommended. Finally, understanding basic statistical concepts such as the Central Limit Theorem also improves the intuition behind why ensembles perform well. Specifically, one would need to look for text on methods of uncertainty quantification. With these resources, a robust understanding of voting ensembles is attainable.
