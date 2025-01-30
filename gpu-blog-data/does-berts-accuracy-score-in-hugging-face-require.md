---
title: "Does BERT's accuracy score in Hugging Face require all labels to be correctly predicted for a prediction to count as correct?"
date: "2025-01-30"
id: "does-berts-accuracy-score-in-hugging-face-require"
---
BERT's accuracy calculation within Hugging Face transformers does not mandate that *all* labels in a multi-label classification scenario must be perfectly matched for a prediction to be considered correct. Instead, the accuracy metric used, by default and often configured in training loops, evaluates the *subset accuracy* for single-label tasks and computes the average accuracy across each predicted label individually for multi-label scenarios. This difference is crucial and a common point of confusion when transitioning from binary or single-class problems to multi-label ones. My experience building a multi-intent classifier for a customer service chatbot using a pre-trained BERT model highlighted this nuance quite significantly during initial evaluation.

The core principle is that the evaluation is performed *per label* rather than *per instance* in multi-label contexts. In a single-label scenario, if you have an instance with ground truth label `[2]` and the prediction is `[2]`, it's considered a correct prediction. Conversely, if the prediction were `[1]`, it would be considered an incorrect prediction. Accuracy is simply the proportion of correct predictions over the total number of predictions. However, this simple accuracy logic does not extend to multi-label classification directly. If your ground truth is `[0, 1, 0]` and you predict `[0, 1, 1]`, the notion of the overall prediction being completely correct is not what determines the accuracy score.

Instead, when using metrics like `accuracy_score` directly or when Hugging Face Trainer computes accuracy internally, it typically calculates the accuracy for *each* label separately and then averages the result. In a simplified view, the calculation can be broken down into three steps:

1.  **Per-Label Comparisons**: For each label position (or column, in a matrix representation), compare the predicted value with the actual value in the corresponding label’s ground truth. In the example above with a ground truth of `[0, 1, 0]` and a prediction of `[0, 1, 1]`, the comparison will occur for each label separately: for the first label (position 0), the predicted and actual values are both 0, which is correct; for the second label (position 1), both are 1, so it is correct; for the third label (position 2) the predicted value is 1, and the actual value is 0, which is incorrect.

2.  **Label-Specific Accuracy**: For each label column, count the number of correct predictions and divide by the total number of instances. Continuing with the example and assuming that this was the only instance, for the first column, we had 1 correct out of 1 instance, therefore the label-specific accuracy is 1. For the second column, also 1 out of 1 correct, therefore the label-specific accuracy is 1. And for the third column, 0 out of 1, meaning the label-specific accuracy is 0.

3.  **Average Accuracy**: Average the label-specific accuracies across all labels. In the example, this yields an average accuracy of (1 + 1 + 0)/3 = 0.6667. Thus, even though the prediction was not a *perfect* match to the labels, the accuracy score reflects which labels it did match successfully.

This averaging method, though providing a singular accuracy score, needs to be interpreted within this multi-label context. It is not reflective of whether an entire instance is correct, but rather provides a measure of performance across each classification task independently. This distinction is not just theoretical; it has profound implications for how one tunes and interprets the results of multi-label BERT models.

To illustrate further, consider three hypothetical training batch outputs with associated ground truths, assuming a multi-label classification task with three labels, processed with a suitable tokenizer and BERT-based model.

**Code Example 1: Demonstrating Accuracy Calculation Manually (Python)**

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Hypothetical predictions from a model (batch size 2)
predictions = np.array([[0, 1, 1], [1, 0, 0]])
# Hypothetical ground truth labels
ground_truth = np.array([[0, 1, 0], [1, 1, 0]])

# Calculate accuracy for each label individually
label_accuracies = []
for i in range(predictions.shape[1]):
    label_acc = accuracy_score(ground_truth[:, i], predictions[:, i])
    label_accuracies.append(label_acc)

# Average the accuracies across labels
average_accuracy = np.mean(label_accuracies)

print("Per-Label Accuracies:", label_accuracies)
print("Average Accuracy:", average_accuracy)
```

In this example, we manually loop through each of the label columns using a Numpy array slice (`[:,i]`) and calculate the accuracy score from sklearn using the `accuracy_score` function. The resulting `label_accuracies` list provides the per-label accuracy, while `average_accuracy` aggregates these per-label scores to provide an overall accuracy score. This code mirrors how a multi-label accuracy is calculated behind the scenes by Hugging Face Trainer or when using metric evaluators directly.

**Code Example 2: Using Hugging Face Trainer with Accuracy Metric**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Mock dataset creation
labels_list = [[0,1,0], [1,0,1], [0,0,1], [1,1,0]]
texts = ["text1", "text2", "text3", "text4"]
dataset_dict = {"labels":labels_list, "text": texts}
dataset = Dataset.from_dict(dataset_dict)

# Tokenizer and model loading
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, problem_type="multi_label_classification")

# Function for tokenizing the text in a batch
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up for Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (logits > 0).astype(int) # Sigmoid activation followed by threshold
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_eval_batch_size=2,
    num_train_epochs=1,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    compute_metrics=compute_metrics
)
# Perform evaluation
trainer.evaluate()
```

This example illustrates the integrated use of `Trainer` in Hugging Face for multi-label classification. The `compute_metrics` function shows the thresholding of logits to produce binary predictions and then it shows how `accuracy_score` is called directly for calculating a single accuracy value over the multi-label instance. The results here won't match example 1 exactly as it is not processed in the same way for ease of demonstration, but the principle still holds.  In actuality, the `Trainer` internally performs the per-label accuracy calculation and then averages them.

**Code Example 3: Demonstrating the averaging of per-label accuracy**

```python
import numpy as np
from sklearn.metrics import accuracy_score

# Hypothetical model output
logits = np.array([
    [[0.2, 0.8, 0.1], [0.9, 0.3, 0.2]], # First Batch of logits
    [[0.1, 0.6, 0.7], [0.3, 0.2, 0.9]]  # Second Batch of logits
    ])

# Ground truth corresponding to the above
labels = np.array([
    [[0,1,0], [1,0,0]], # First batch of labels
    [[0,1,1], [0,0,1]]  # Second batch of labels
])

batch_size = logits.shape[0]
num_labels = logits.shape[2]
label_accuracies = []

for label_idx in range(num_labels):
    all_predictions = []
    all_ground_truth = []
    for batch_idx in range(batch_size):
        batch_logits = logits[batch_idx]
        batch_labels = labels[batch_idx]
        predictions = (batch_logits > 0.5).astype(int) # thresholding
        all_predictions.extend(predictions[:,label_idx]) # selecting the label
        all_ground_truth.extend(batch_labels[:,label_idx]) # selecting the label
    label_accuracy = accuracy_score(all_ground_truth, all_predictions) # computes accuracy for a given label
    label_accuracies.append(label_accuracy) # collecting per-label accuracies

avg_accuracy = np.mean(label_accuracies) # Average per-label accuracies
print("Per-Label Accuracies:", label_accuracies)
print("Average Accuracy:", avg_accuracy)
```

This example takes raw model logits and ground truths as they might appear after processing and demonstrates the process of individually thresholding the logits per batch and per label, using a manual loop. The per-label accuracies are computed and then averaged into a final score, this shows the detail of how the aggregation takes place.

For further understanding, I recommend exploring resources such as the Hugging Face documentation on evaluation metrics, and the scikit-learn documentation, particularly the sections on accuracy score for multi-label scenarios. Further investigation into confusion matrix analysis, precision, recall, and F1-score for multi-label classification provides a more comprehensive understanding of model behavior and performance.
