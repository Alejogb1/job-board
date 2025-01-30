---
title: "Why is the Bi-LSTM achieving such a low macro-average F1 score in sequence labeling?"
date: "2025-01-30"
id: "why-is-the-bi-lstm-achieving-such-a-low"
---
The persistent low macro-average F1 score observed in my recent sequence labeling experiments using a Bidirectional Long Short-Term Memory (Bi-LSTM) network points directly to an issue with the model's handling of infrequent, yet critical, labels within the dataset. I’ve observed this pattern before in scenarios involving imbalanced datasets, and it frequently reveals a misalignment between the model's optimization process and the desired evaluation metric.

The crux of the problem isn't necessarily that the Bi-LSTM is failing to learn; rather, it's that the model optimizes to maximize the overall accuracy or loss, which are heavily influenced by the dominant labels. Macro-average F1 score, on the other hand, treats all labels equally, regardless of their prevalence. Consequently, the model prioritizes learning to recognize the frequent labels with high precision and recall, often at the expense of the infrequent ones. These underrepresented labels are then misclassified or missed entirely, resulting in low precision and recall scores for these classes, and a subsequently depressed macro-average F1 score. This phenomenon highlights the importance of evaluation metrics beyond simple accuracy in imbalanced classification tasks.

My experience with sequence labeling projects has taught me that understanding the dataset's label distribution is paramount before selecting evaluation metrics or model architectures. Bi-LSTMs, by their nature, excel at capturing temporal dependencies, and perform well when given a balanced, well-represented dataset. However, when presented with an imbalanced label distribution, the underlying optimization mechanics lead to suboptimal performance with respect to the macro-averaged F1 score. The total loss calculation, typically the mean of losses over all time steps and batch instances, will be dominated by frequent labels, causing the model to allocate more learning capacity to these more common categories.

To better understand this behavior and explore remedies, let’s examine some hypothetical, but representative scenarios. Consider a Named Entity Recognition (NER) task, where the labels are B-ORG, I-ORG, B-PER, I-PER, B-LOC, I-LOC, and ‘O’ (outside any named entity). In most real-world text corpora, the ‘O’ tag will typically dominate, often constituting over 80% of all labels. This dominance will bias a standard Bi-LSTM training loop. Furthermore, certain named entity types like “ORG” or “LOC” might be substantially more frequent than “PER”. This imbalance directly affects the model's ability to accurately identify the less frequent labels, ultimately hurting the macro-averaged F1 score.

Now, let's examine some code examples and their implications in this context. I'll present these examples using PyTorch, as it’s a library I frequently use for my deep learning work. These examples will illustrate how the imbalance affects the training, and how we might try to address it.

**Example 1: Baseline Bi-LSTM Training with Cross-Entropy Loss**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume 'vocab_size', 'embedding_dim', 'hidden_size', and 'num_labels' are defined
# Assume 'train_data', 'train_labels' are preprocessed and ready for input.
# This example assumes an imbalanced label distribution

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_labels):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        return output

model = BiLSTMModel(vocab_size, embedding_dim, hidden_size, num_labels)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(train_data)
    loss = criterion(predictions.view(-1, num_labels), train_labels.view(-1)) # Flatten for loss calculation
    loss.backward()
    optimizer.step()
```

This initial example sets up a basic Bi-LSTM sequence labeling model. The critical part is the use of `nn.CrossEntropyLoss`. This loss function implicitly assumes balanced class distributions and optimizes to minimize the average loss, which as we've established, favors frequent labels.  A low macro-average F1 score often occurs despite the training loss decreasing, because the model is still not performing well on the minority classes. The imbalanced nature is not specifically addressed in the training process.

**Example 2: Class-Weighted Loss**

```python
# Assume 'class_weights' are calculated based on the frequency of labels
# Example: class_weights = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
class_weights = get_class_weights(train_labels) #A separate function would calculate class_weights

criterion_weighted = nn.CrossEntropyLoss(weight=class_weights) #Weighted CrossEntropyLoss

#Training Loop remains the same with the new criterion
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(train_data)
    loss = criterion_weighted(predictions.view(-1, num_labels), train_labels.view(-1))
    loss.backward()
    optimizer.step()
```

This modification introduces class weighting to the `CrossEntropyLoss` function. By using `nn.CrossEntropyLoss` with the `weight` parameter, we penalize errors on infrequent classes more heavily. The weights are inversely proportional to the label frequencies; less frequent labels have higher weights, so errors are penalized more.  This helps to balance the contribution of each label to the loss, nudging the model to pay greater attention to the underrepresented labels. This can improve the model’s performance with respect to the macro-averaged F1 score, since the improvement of the model on the minority classes improves the scores and also gives more importance to improvement in these classes during optimization.

**Example 3: Focal Loss**

```python
# Focal loss implementation (simplification)
class FocalLoss(nn.Module):
  def __init__(self, gamma=2, alpha=None, reduction="mean"):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction

  def forward(self, logits, targets):
    pt = torch.softmax(logits, dim=-1)
    pt = torch.gather(pt, dim=-1, index = targets.unsqueeze(-1))
    pt = pt.squeeze(-1)
    loss = - self.alpha * (1 - pt) ** self.gamma * torch.log(pt) if self.alpha is not None else -(1-pt)**self.gamma * torch.log(pt)

    if self.reduction == "mean":
      return torch.mean(loss)
    elif self.reduction == "sum":
        return torch.sum(loss)
    else:
        return loss

criterion_focal = FocalLoss(gamma=2, alpha=None) #Focal Loss criterion
#Training loop with new criterion.
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(train_data)
    loss = criterion_focal(predictions.view(-1, num_labels), train_labels.view(-1))
    loss.backward()
    optimizer.step()
```

Here, I've implemented a simplified version of Focal Loss, a method that further addresses class imbalance by reducing the loss contribution from well-classified examples (both positive and negative). The key parameter, gamma, adjusts the rate at which the loss contribution of easily classified samples is reduced. This can help the model focus on hard-to-classify samples and improve performance on less frequent classes. When gamma is greater than 0, the loss of the well-classified samples decreases compared to cross-entropy. Alpha is a balancing parameter similar to class weights, used to weigh the loss for the minority and the majority class differently. This method focuses learning more on difficult examples that are frequently associated with minority classes. This improves the performance with respect to the Macro-F1 score.

Based on my experience, when encountering such issues with macro-averaged F1 score, consider these approaches:

1. **Data Augmentation:** If possible, augment the dataset specifically with examples that contain less frequent classes. This should be done carefully to avoid artificially introducing noise.
2. **Oversampling/Undersampling:** Oversampling minority classes or undersampling majority classes in the training data can help balance the training dataset. This is important when data augmentation cannot be done effectively.
3. **Adjust Evaluation Metrics:** Use appropriate metrics such as precision, recall and F1 scores that are calculated for each class independently, in addition to the macro-average. This helps identify the exact source of the low macro-average F1 score. If the performance for a specific class is too low, a directed solution for that specific problem may be required.
4. **Ensemble Methods:** Combining predictions from multiple models trained with different subsets of the data or different configurations can sometimes improve performance on rare classes.
5. **Careful Hyperparameter Tuning:** Tune hyperparameters for your model and loss function specifically when class imbalances exist.

For further exploration, I recommend delving into resources on handling imbalanced datasets, such as research papers on cost-sensitive learning, and practical guides to techniques like focal loss and class weighting. Additionally, reviewing literature on sequence labeling and evaluation methods specific to natural language processing will provide a deeper theoretical understanding of these concepts. Exploring code repositories that implement advanced loss functions will help in practical implementations and further understanding. Finally, focus on the theory behind the loss functions and the implications of the gradient calculation on parameter updates.
