---
title: "How can I interpret precision, recall, and F1-score in a custom SpaCy tagger?"
date: "2024-12-23"
id: "how-can-i-interpret-precision-recall-and-f1-score-in-a-custom-spacy-tagger"
---

, let’s get into this. Instead of jumping straight into formulas, let’s consider the practical implications first, something that's been baked into my brain from a few past projects. Specifically, I remember one involved a highly specific named entity recognition task for legal documents. We needed to identify particular clauses and document types with extremely high accuracy, where misclassifying a “contract amendment” as just “contract” could have serious consequences. This isn’t a playground; this is about ensuring the robustness of models we deploy.

Now, regarding precision, recall, and the F1-score, these are foundational metrics used to evaluate the performance of classification models, including the custom spaCy tagger you're working with. They are particularly important when you're not just looking at overall accuracy (which can be misleading, especially with imbalanced datasets). Each metric provides a unique lens on the model’s performance, and understanding them separately is vital to improving your model.

Let's unpack each one.

**Precision**:

Precision answers the question: of all the entities the model *said* were a certain type, how many actually *were* that type? It measures the proportion of true positives out of all predicted positives. Formally, it’s calculated as:

`Precision = True Positives / (True Positives + False Positives)`

In my experience, precision becomes extremely critical when false positives have high costs. In the legal document project I mentioned, a false positive – tagging a random phrase as a high-stakes clause – would trigger an unnecessary workflow, introducing inefficiency. A high precision means that when our tagger predicts a specific tag, we can be very confident that it's correct, reducing these wasteful errors.

**Recall**:

Recall focuses on: of all the entities that *actually* were of a certain type, how many did the model correctly identify? It measures the proportion of true positives out of all the actual positives. The formula is:

`Recall = True Positives / (True Positives + False Negatives)`

Recall is paramount when it’s more costly to *miss* an entity than to misclassify others. Back to the legal documents, a false negative (missing an actual contract amendment) would be far worse than misclassifying something as a contract amendment that wasn’t. A high recall indicates that our model is good at identifying the majority of the actual occurrences of a particular entity.

**F1-Score**:

The F1-score is the harmonic mean of precision and recall, calculated as:

`F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`

The F1-score aims to balance precision and recall. It's most useful when you want to find a single metric that gives a sense of the model’s overall performance, especially when both precision and recall are important. We often used the F1-score to select between multiple models or parameter configurations in that legal project.

Now, let’s illustrate this with some code. I’ll use Python to demonstrate calculations within the context of tag classification, simulating the kind of evaluation I often build in real projects:

```python
def calculate_metrics(true_labels, predicted_labels, target_label):
    """Calculates precision, recall, and F1-score for a specific label.

    Args:
        true_labels: List of ground truth labels.
        predicted_labels: List of model's predicted labels.
        target_label: The label to evaluate.

    Returns:
        A tuple containing precision, recall, and f1-score.
    """

    true_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == target_label and pred == target_label)
    false_positives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true != target_label and pred == target_label)
    false_negatives = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == target_label and pred != target_label)

    if (true_positives + false_positives) == 0:
        precision = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)

    if (true_positives + false_negatives) == 0:
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if (precision + recall) == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


true_labels = ["CONTRACT", "AMENDMENT", "CONTRACT", "CLAUSE", "AMENDMENT", "CONTRACT"]
predicted_labels = ["CONTRACT", "CONTRACT", "CONTRACT", "CLAUSE", "AMENDMENT", "CLAUSE"]

precision_amendment, recall_amendment, f1_amendment = calculate_metrics(true_labels, predicted_labels, "AMENDMENT")
precision_contract, recall_contract, f1_contract = calculate_metrics(true_labels, predicted_labels, "CONTRACT")
precision_clause, recall_clause, f1_clause = calculate_metrics(true_labels, predicted_labels, "CLAUSE")


print(f"AMENDMENT - Precision: {precision_amendment:.2f}, Recall: {recall_amendment:.2f}, F1: {f1_amendment:.2f}")
print(f"CONTRACT  - Precision: {precision_contract:.2f}, Recall: {recall_contract:.2f}, F1: {f1_contract:.2f}")
print(f"CLAUSE - Precision: {precision_clause:.2f}, Recall: {recall_clause:.2f}, F1: {f1_clause:.2f}")

# Example of calculating metrics using sklearn.metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision_amendment_sk = precision_score(true_labels, predicted_labels, average=None, labels=["AMENDMENT"])[0]
recall_amendment_sk = recall_score(true_labels, predicted_labels, average=None, labels=["AMENDMENT"])[0]
f1_amendment_sk = f1_score(true_labels, predicted_labels, average=None, labels=["AMENDMENT"])[0]

print(f"\nAMENDMENT (sklearn) - Precision: {precision_amendment_sk:.2f}, Recall: {recall_amendment_sk:.2f}, F1: {f1_amendment_sk:.2f}")


```

This code shows how to calculate these metrics from raw label predictions. You can compare it to more polished results using library functions.

```python
import spacy

def evaluate_spacy_tagger(nlp, test_data):
    """Evaluates a spaCy tagger.

       Args:
           nlp:  The trained spaCy model.
           test_data: List of tuples, each containing text and a dict of gold labels.

       Returns:
         A dictionary of calculated metrics.

    """
    from collections import defaultdict
    true_labels = []
    predicted_labels = []
    for text, gold_labels in test_data:
        doc = nlp(text)
        true_labels_doc = []
        predicted_labels_doc = []

        # Assuming gold_labels is in the form {"entities": [(start, end, label), ...]}
        gold_entities = gold_labels.get('entities', [])
        for start, end, label in gold_entities:
            true_labels_doc.append((start, end, label))
        
        for token in doc:
            if token.ent_type_:
                predicted_labels_doc.append((token.idx, token.idx + len(token), token.ent_type_ ))
        
        true_labels.extend(true_labels_doc)
        predicted_labels.extend(predicted_labels_doc)
        
    labels = set([label for _, _, label in true_labels])

    metrics = defaultdict(dict)
    for label in labels:
        true_l = [1 if (start,end, l) ==  (start_true, end_true,l_true) else 0 for start_true, end_true, l_true in true_labels for start, end, l in predicted_labels if start_true == start and end_true ==end]
        predicted_l = [1 if (start,end, l) ==  (start_pred, end_pred,l_pred) else 0 for start_pred, end_pred, l_pred in predicted_labels for start, end, l in true_labels if start_pred == start and end_pred ==end]
       
        if true_l and predicted_l:
           precision, recall, f1 = calculate_metrics(true_l, predicted_l, 1)
           metrics[label]["precision"] = precision
           metrics[label]["recall"] = recall
           metrics[label]["f1"] = f1
        else:
           metrics[label]["precision"] = 0.0
           metrics[label]["recall"] = 0.0
           metrics[label]["f1"] = 0.0
        
    return metrics


# Example usage (replace with your real model and test data):
nlp = spacy.load("en_core_web_sm") # or your custom model
test_data = [
    ("This is a contract amendment.", {"entities": [(11, 19, "CONTRACT"), (20, 29, "AMENDMENT")]}),
    ("Another clause in the agreement.", {"entities": [(8, 14, "CLAUSE")]}),
    ("Contract number 123.", {"entities": [(0,8, "CONTRACT")]}),
]

results = evaluate_spacy_tagger(nlp, test_data)
for label, values in results.items():
    print(f"Label: {label}, Precision: {values['precision']:.2f}, Recall: {values['recall']:.2f}, F1: {values['f1']:.2f}")
```

This example illustrates how to use the function with a spaCy model. Remember, you’d integrate this directly into your evaluation pipelines.

Lastly, here's a brief function to demonstrate how these metrics can be used in the context of training.

```python
def log_metrics_during_training(epoch, precision_dict, recall_dict, f1_dict):
    """Logs metrics during training process

    Args:
        epoch (int): current training epoch
        precision_dict (dict): precision scores dictionary
        recall_dict (dict): recall scores dictionary
        f1_dict (dict): f1-scores dictionary

    """
    print(f"--- Epoch {epoch} Metrics ---")
    for label in precision_dict.keys():
        print(f"Label: {label}, Precision: {precision_dict[label]:.2f}, Recall: {recall_dict[label]:.2f}, F1: {f1_dict[label]:.2f}")

# Usage in a training loop
# during training you'd calculate the metrics, as in the previous examples and
# then you'd pass them to the logging function in each epoch

# example of metrics dictionaries
precision_example = {"CONTRACT": 0.85, "AMENDMENT": 0.72, "CLAUSE": 0.91}
recall_example = {"CONTRACT": 0.88, "AMENDMENT": 0.68, "CLAUSE": 0.90}
f1_example = {"CONTRACT": 0.86, "AMENDMENT": 0.70, "CLAUSE": 0.90}


log_metrics_during_training(10, precision_example, recall_example, f1_example)
```

This demonstrates how metrics can be logged during training.

For deeper understanding, I would highly recommend digging into "Speech and Language Processing" by Daniel Jurafsky and James H. Martin. Specifically, the chapter on classification evaluation will provide an excellent foundation. Another valuable resource is "Information Retrieval: Implementing and Evaluating Search Engines" by Stefan Büttcher, Charles L. A. Clarke, and Gordon V. Cormack; while focusing on search, it delves deeply into evaluation metrics that apply equally to taggers. Finally, for a practical approach, you might look at research papers from conferences like ACL, EMNLP or NAACL which often have detailed experiments on named entity recognition and similar tasks. These are a great source of real world evaluation examples.

Remember, these metrics are tools that help you understand and debug your tagger. It's not about aiming for perfect scores but about understanding the trade-offs and how they affect the downstream application.
