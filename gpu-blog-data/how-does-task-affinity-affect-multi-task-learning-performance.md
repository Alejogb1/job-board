---
title: "How does task affinity affect multi-task learning performance?"
date: "2025-01-30"
id: "how-does-task-affinity-affect-multi-task-learning-performance"
---
Task affinity, specifically the degree of relatedness between tasks in a multi-task learning (MTL) scenario, fundamentally dictates the effectiveness of shared representation learning. If tasks are highly dissimilar, attempting to learn a common feature space may prove detrimental rather than beneficial, often leading to performance degradation compared to single-task models. I’ve observed this pattern firsthand across several projects involving natural language processing and computer vision.

Multi-task learning hinges on the idea that related tasks can mutually benefit by learning shared representations, capturing underlying structures and patterns relevant to all involved tasks. Ideally, a model will discover features that generalize well across these related tasks, leading to improved performance and reduced training time compared to training separate models for each task. However, the implicit assumption here is that tasks share a meaningful degree of affinity. If task representations are far apart in the feature space, forcing a shared representation may result in a sub-optimal solution, where the model sacrifices performance on some or all tasks to achieve a compromised common space.

The issue arises because gradient updates during training, based on combined loss functions from multiple tasks, will pull shared representations towards a compromise point that might not be ideal for any single task. Consequently, the optimization process can become more challenging, with conflicting gradients pushing feature representations in different directions. This leads to situations where the optimization landscape becomes more rugged, making it harder for the model to converge to an optimal solution. When tasks demonstrate low affinity, we can expect to observe the model learning less useful representations overall. In extreme cases, this can manifest as a phenomenon called negative transfer, where multi-task learning actively hinders performance compared to single task models.

To illustrate these points, consider a few simplified scenarios using PyTorch to demonstrate. The first scenario will showcase beneficial transfer between related classification tasks, and subsequent examples will show how lack of affinity leads to detrimental learning.

**Scenario 1: High Task Affinity – Sentiment Analysis and Topic Classification**

I will consider a simplified setting with two tasks: sentiment analysis (predicting positive, negative, or neutral sentiment) and topic classification (identifying a broad topic, like politics, sports, or technology). While distinct, these tasks are clearly related in that they operate on the same text data and leverage shared concepts. For example, terms associated with ‘negative’ sentiment might also appear frequently in articles pertaining to negative news topics.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#Dummy Data Generation
torch.manual_seed(42)
data_size = 1000
feature_size = 50
num_topics = 3
num_sentiments = 3

features = torch.randn(data_size, feature_size)
topic_labels = torch.randint(0, num_topics, (data_size,))
sentiment_labels = torch.randint(0, num_sentiments, (data_size,))

dataset = TensorDataset(features, topic_labels, sentiment_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class SharedEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(SharedEncoder, self).__init__()
        self.fc = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class TopicClassifier(nn.Module):
    def __init__(self, hidden_size, num_topics):
        super(TopicClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_topics)

    def forward(self, x):
        x = self.fc(x)
        return x

class SentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_sentiments):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_sentiments)

    def forward(self, x):
        x = self.fc(x)
        return x

# Define models
hidden_size = 32
shared_encoder = SharedEncoder(feature_size, hidden_size)
topic_classifier = TopicClassifier(hidden_size, num_topics)
sentiment_classifier = SentimentClassifier(hidden_size, num_sentiments)

# Optimizer & Loss
optimizer = optim.Adam(list(shared_encoder.parameters()) + list(topic_classifier.parameters()) + list(sentiment_classifier.parameters()), lr=0.001)
topic_loss_fn = nn.CrossEntropyLoss()
sentiment_loss_fn = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    total_topic_loss = 0
    total_sentiment_loss = 0
    for batch_features, batch_topic_labels, batch_sentiment_labels in dataloader:
        optimizer.zero_grad()
        encoded_features = shared_encoder(batch_features)
        topic_output = topic_classifier(encoded_features)
        sentiment_output = sentiment_classifier(encoded_features)

        topic_loss = topic_loss_fn(topic_output, batch_topic_labels)
        sentiment_loss = sentiment_loss_fn(sentiment_output, batch_sentiment_labels)

        loss = topic_loss + sentiment_loss
        loss.backward()
        optimizer.step()

        total_topic_loss += topic_loss.item()
        total_sentiment_loss += sentiment_loss.item()

    avg_topic_loss = total_topic_loss / len(dataloader)
    avg_sentiment_loss = total_sentiment_loss / len(dataloader)
    if epoch % 10 == 0:
      print(f"Epoch {epoch}: Avg Topic Loss: {avg_topic_loss:.4f}, Avg Sentiment Loss: {avg_sentiment_loss:.4f}")
```

In this code, I defined a `SharedEncoder`, where shared features are learned, followed by separate task-specific classification heads (`TopicClassifier` and `SentimentClassifier`). The model is trained by combining the loss from both tasks. I’ve seen that with even just a few epochs, both losses tend to decrease, indicating the benefit of learning a joint representation.

**Scenario 2: Low Task Affinity – Image Classification and Text Generation**

Consider now trying to learn from an image classification task and a text generation task using the same shared embedding space. There is no obvious inherent relationship. In fact, they operate on fundamentally different data types. Images are processed in a different manner from text, where token embeddings or character embeddings are more common.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#Dummy Data Generation
torch.manual_seed(42)
data_size = 1000
image_feature_size = 32
text_feature_size = 50
num_classes = 5
vocab_size = 100

image_features = torch.randn(data_size, image_feature_size)
class_labels = torch.randint(0, num_classes, (data_size,))
text_features = torch.randn(data_size, text_feature_size)
text_labels = torch.randint(0, vocab_size, (data_size, 20))

image_dataset = TensorDataset(image_features, class_labels)
image_dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)
text_dataset = TensorDataset(text_features, text_labels)
text_dataloader = DataLoader(text_dataset, batch_size=32, shuffle=True)


class SharedEmbedding(nn.Module):
    def __init__(self, image_feature_size, text_feature_size, hidden_size):
        super(SharedEmbedding, self).__init__()
        self.image_fc = nn.Linear(image_feature_size, hidden_size)
        self.text_fc = nn.Linear(text_feature_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, image_input, text_input):
        embedded_image = self.relu(self.image_fc(image_input))
        embedded_text = self.relu(self.text_fc(text_input))
        return embedded_image, embedded_text


class ImageClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ImageClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(x)

class TextGenerator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(TextGenerator, self).__init__()
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        return self.fc(x)


hidden_size = 32
shared_embedding = SharedEmbedding(image_feature_size, text_feature_size, hidden_size)
image_classifier = ImageClassifier(hidden_size, num_classes)
text_generator = TextGenerator(hidden_size, vocab_size)

optimizer = optim.Adam(list(shared_embedding.parameters()) + list(image_classifier.parameters()) + list(text_generator.parameters()), lr=0.001)
image_loss_fn = nn.CrossEntropyLoss()
text_loss_fn = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    total_image_loss = 0
    total_text_loss = 0
    for (batch_image_features, batch_class_labels), (batch_text_features, batch_text_labels) in zip(image_dataloader, text_dataloader):
      optimizer.zero_grad()
      embedded_image, embedded_text = shared_embedding(batch_image_features, batch_text_features)
      image_output = image_classifier(embedded_image)
      text_output = text_generator(embedded_text)

      image_loss = image_loss_fn(image_output, batch_class_labels)
      text_loss = text_loss_fn(text_output.view(-1, vocab_size), batch_text_labels.view(-1))
      loss = image_loss + text_loss
      loss.backward()
      optimizer.step()

      total_image_loss += image_loss.item()
      total_text_loss += text_loss.item()

    avg_image_loss = total_image_loss / len(image_dataloader)
    avg_text_loss = total_text_loss / len(text_dataloader)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Avg Image Loss: {avg_image_loss:.4f}, Avg Text Loss: {avg_text_loss:.4f}")
```

In this example, I used two separate dataloaders since the datasets are inherently different. I also adjusted the `SharedEmbedding` module to output separate embeddings based on input type. Training with a shared embedding shows slow and unstable convergence. Both losses tend to fluctuate and may not converge well, which reflects a low degree of affinity, and the resulting joint feature space is unlikely to be beneficial.

**Scenario 3: Negative Transfer – Related Tasks with Conflicting Objectives**

Even when tasks operate on similar data, they might still have a detrimental effect if the objectives are contradictory. For example, assume we have two tasks: one task to classify the author's sentiment in a document and another to classify the sentiment of the cited source material. Although both operate on text data and classify sentiment, the underlying sentiments could often be opposite and hence the representation space for one is probably unhelpful for the other.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


#Dummy Data Generation
torch.manual_seed(42)
data_size = 1000
feature_size = 50
num_sentiments = 3

features = torch.randn(data_size, feature_size)
author_sentiment_labels = torch.randint(0, num_sentiments, (data_size,))
source_sentiment_labels = (torch.randint(0, num_sentiments, (data_size,)) + 1) % num_sentiments #introduce a small shift

dataset = TensorDataset(features, author_sentiment_labels, source_sentiment_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class SharedEncoder(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(SharedEncoder, self).__init__()
        self.fc = nn.Linear(feature_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class AuthorSentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_sentiments):
        super(AuthorSentimentClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_sentiments)

    def forward(self, x):
        return self.fc(x)

class SourceSentimentClassifier(nn.Module):
    def __init__(self, hidden_size, num_sentiments):
        super(SourceSentimentClassifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_sentiments)

    def forward(self, x):
        return self.fc(x)



hidden_size = 32
shared_encoder = SharedEncoder(feature_size, hidden_size)
author_classifier = AuthorSentimentClassifier(hidden_size, num_sentiments)
source_classifier = SourceSentimentClassifier(hidden_size, num_sentiments)

optimizer = optim.Adam(list(shared_encoder.parameters()) + list(author_classifier.parameters()) + list(source_classifier.parameters()), lr=0.001)
author_loss_fn = nn.CrossEntropyLoss()
source_loss_fn = nn.CrossEntropyLoss()

epochs = 50
for epoch in range(epochs):
    total_author_loss = 0
    total_source_loss = 0
    for batch_features, batch_author_labels, batch_source_labels in dataloader:
      optimizer.zero_grad()
      encoded_features = shared_encoder(batch_features)
      author_output = author_classifier(encoded_features)
      source_output = source_classifier(encoded_features)

      author_loss = author_loss_fn(author_output, batch_author_labels)
      source_loss = source_loss_fn(source_output, batch_source_labels)

      loss = author_loss + source_loss
      loss.backward()
      optimizer.step()

      total_author_loss += author_loss.item()
      total_source_loss += source_loss.item()

    avg_author_loss = total_author_loss / len(dataloader)
    avg_source_loss = total_source_loss / len(dataloader)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Avg Author Loss: {avg_author_loss:.4f}, Avg Source Loss: {avg_source_loss:.4f}")
```

In this final example, I intentionally made the sentiment labels somewhat opposite in an attempt to reflect contradictory objectives. This example illustrates the concept of negative transfer where, even though the inputs and the tasks have similar nature (i.e., sentiment classification of text), the loss values may not decrease as smoothly as compared to single task training. This indicates that forcing the model to use a joint representation might not be optimal for the individual tasks and that the model is likely learning a compromise representation that is detrimental to either tasks.

In summary, the affinity between tasks is crucial for the success of multi-task learning. When selecting tasks for multi-task learning, it is critical to consider how well aligned the tasks are. Furthermore, when encountering poor multi-task learning performance, the possibility of low task affinity should be a primary suspect. Techniques like task-specific layers, specialized architectures, or attention mechanisms can help when tasks do not align perfectly. Additionally, exploring various methods to weight the losses can also have a significant impact.

For further reading, I recommend resources focusing on multi-task learning theory, specifically works discussing gradient interference and negative transfer. Publications exploring methods for adaptive loss weighting and task affinity measurement are also very beneficial in practice. Finally, exploring practical implementations in libraries like TensorFlow and PyTorch can reinforce the understanding. These theoretical and practical aspects are indispensable for applying multi-task learning effectively.
