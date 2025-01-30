---
title: "Can BERT-large perform text classification without fine-tuning?"
date: "2025-01-30"
id: "can-bert-large-perform-text-classification-without-fine-tuning"
---
A pre-trained BERT-large model, while possessing a deep understanding of language, requires specific modifications to function effectively as a text classifier; direct zero-shot classification, while conceptually possible, rarely achieves acceptable performance without targeted adaptation. My experience developing several NLP applications has consistently highlighted this limitation. BERT-large's primary function during pre-training is to generate contextualized word embeddings and internal representations, not to predict specific category labels. Therefore, its architecture needs to be adjusted for this task.

The foundational challenge lies in BERT-large's output. The core model produces sequences of hidden state vectors, one for each input token. These vectors encode the complex nuances of the input sentence but are not directly interpretable as class probabilities. Text classification requires a prediction layer that transforms these internal representations into a vector representing the likelihood of each potential class. Without this layer, BERT-large cannot associate its language understanding with concrete categories. Furthermore, the original pre-training objective (masked language modeling and next sentence prediction) is fundamentally different from text classification. These tasks optimize for general language understanding, while classification requires learning relationships between textual input and specific categories. This distinction in objectives explains why direct application of the pre-trained model is suboptimal for classification.

Zero-shot, or more accurately, few-shot transfer learning, is possible with BERT by employing clever prompts and utilizing the model’s inherent linguistic capabilities. However, these methods should be distinguished from true ‘zero-shot’ classification. It is not that the model is performing classification without learning any association between text and categories, but that this association is indirectly inferred, not directly trained. This typically involves providing the model with instructions in the form of templates that implicitly define a classification task. For example, you might input a text: "The movie was utterly fantastic. Sentiment: [MASK]". The BERT model then fills the [MASK] position, with a preference to predict tokens indicating sentiment like 'positive'. However, the accuracy and reliability of this method are highly sensitive to the prompt engineering and generally lower compared to explicitly fine-tuning. These prompt-based approaches implicitly teach BERT the category associations, although in a more indirect manner.

Here are three specific examples to better illustrate how BERT-large needs to be adapted for text classification. Note that although I will not present actual code implementing a BERT model, the core concepts can be shown through pseudo-code with common libraries in Python such as PyTorch.

**Example 1: Adding a Linear Classification Layer**

In its most straightforward adaptation, a linear layer is appended after the BERT encoder. This layer transforms the output of BERT’s [CLS] token (the special token placed at the beginning of each input sequence which represents the entire sequence) into the desired classification space. In pseudo-code, the process can be expressed as:

```python
# Assume 'bert_model' is the pre-trained BERT-large instance.
# Assume 'input_ids' is the tokenized and numericalized input text.
import torch
import torch.nn as nn

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        cls_output = outputs.last_hidden_state[:, 0, :] # Extract [CLS] output
        logits = self.classifier(cls_output) # Apply linear layer
        return logits

# Example usage
num_classes = 3 # for example, positive, negative, neutral
model = BERTClassifier(bert_model, num_classes)

# Input text, tokenized and converted to numerical IDs
input_ids = torch.tensor([[101, 2023, 2003, 1037, 1020, 2331, 1012, 102]])
logits = model(input_ids) # Output logits (pre-softmax scores)
```

Here, I illustrate the addition of a linear layer (`nn.Linear`) after the base BERT model. This layer takes the contextualized representation of the [CLS] token (accessible via the last hidden state of the sequence at the zeroth position, i.e., `outputs.last_hidden_state[:, 0, :]`) and maps it into a vector of size equal to `num_classes` (e.g., the number of distinct sentiment labels). This `logits` output is then further transformed into probabilities via a softmax activation function before being used for prediction and training. Without this linear transformation, BERT’s output remains a high-dimensional representation unsuitable for direct classification. This is a fundamental requirement for converting the embeddings to class scores.

**Example 2: Utilizing Sequence Pooling**

An alternative to the [CLS] token is sequence pooling. In this technique, the outputs for *all* tokens in the sequence are used to derive a single representation. One common approach is to average these hidden states or use a max-pooling operation. The resulting representation is then passed to the linear classification layer.  Here’s the pseudo-code for such an approach:

```python
import torch
import torch.nn as nn

class BERTClassifierPooling(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifierPooling, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        sequence_output = outputs.last_hidden_state  # Get all token outputs

        pooled_output = torch.mean(sequence_output, dim=1) # Average pooling across tokens

        logits = self.classifier(pooled_output) # Apply linear layer
        return logits

# Example usage, similar to Example 1
num_classes = 3 # positive, negative, neutral
model = BERTClassifierPooling(bert_model, num_classes)
input_ids = torch.tensor([[101, 2023, 2003, 1037, 1020, 2331, 1012, 102]])
logits = model(input_ids)
```

In this example, instead of only relying on the [CLS] token output, we extract all tokens' last hidden state (`outputs.last_hidden_state`), then calculate the mean across the second dimension (the token dimension). The `pooled_output` now represents an aggregated representation of the entire sequence which can be used for classification. Although this method can potentially capture more information than relying solely on [CLS], it usually doesn’t improve the accuracy over the CLS token pooling; sometimes it can even worsen it, depending on the task and the data available. Therefore, its performance should be carefully evaluated. The final classification layer works the same way as Example 1.

**Example 3: Using a Task-Specific Head**

In more advanced scenarios, a more complex task-specific head might be required, beyond a simple linear layer. This could include incorporating non-linear activation functions or multiple fully connected layers. Additionally, attention mechanisms can be added for more sophisticated sequence processing. The specific details of this “head” are highly dependent on the specifics of the classification task. While it is difficult to generalize the implementation, the core idea is to transform the BERT representation into a suitable representation for the task, such as a task-specific embedding, then feed this to the classification layer. This adds more capacity and learns more task-specific representation during fine-tuning, but increases the risk of overfitting. Here is a conceptual outline:

```python
import torch
import torch.nn as nn

class BERTClassifierComplex(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifierComplex, self).__init__()
        self.bert = bert_model
        self.task_specific_head = nn.Sequential(
                nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        cls_output = outputs.last_hidden_state[:, 0, :] # Extract [CLS]

        task_output = self.task_specific_head(cls_output)
        logits = self.classifier(task_output)
        return logits

# Example usage, similar to Example 1
num_classes = 3
model = BERTClassifierComplex(bert_model, num_classes)
input_ids = torch.tensor([[101, 2023, 2003, 1037, 1020, 2331, 1012, 102]])
logits = model(input_ids)
```

In this example, a `task_specific_head` (implemented here as a simple sequential network with a linear transformation, a ReLU activation and a dropout layer) is introduced before the final classification linear layer. This adds complexity and adaptability to the model during fine-tuning and is often better when having a larger amount of training data for the task.

In summary, while the core of BERT-large captures the nuanced relationships between words in a sequence, directly applying its pre-trained outputs to text classification tasks will yield subpar results. You must add task-specific layers and then fine-tune the entire architecture to obtain good performance. The primary reason is that the pre-training task (mask prediction) and the classification task are distinct, resulting in a model not initially set up to perform the classification.

For individuals looking to further their understanding and implementation skills in this area, I recommend delving into resources covering topics such as “Transformer Networks,” "Fine-tuning Language Models,” and “Text Classification.” Specifically, pay close attention to documentation and tutorials for popular deep learning libraries which contain examples of using BERT for text classification tasks. Study research papers on BERT modifications for different downstream tasks. Invest time to explore publicly available pre-trained models and related pre-processing utilities. Also, it is worthwhile to become familiar with the concepts of training and evaluation metrics for classification models. Thorough study and experimentation are vital for effective model deployment.
