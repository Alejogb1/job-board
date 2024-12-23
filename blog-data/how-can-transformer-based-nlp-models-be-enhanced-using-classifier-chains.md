---
title: "How can transformer-based NLP models be enhanced using classifier chains?"
date: "2024-12-23"
id: "how-can-transformer-based-nlp-models-be-enhanced-using-classifier-chains"
---

,  I remember back in '19, working on a particularly thorny multi-label classification problem in a customer feedback analysis pipeline. We were wrestling, or rather, *engaging* with the limitations of standard transformer outputs when faced with multiple interrelated labels. That’s when I really started to appreciate the power of classifier chains, and their potential to enhance these transformer models, not as a replacement but as a critical augmentation.

The core challenge with direct multi-label classification using transformers often boils down to ignoring the inherent dependencies between labels. We typically output a sigmoid activated vector of logits, representing the probability of each label being present. This treats labels as completely independent events, which, in reality, they rarely are. Classifier chains, on the other hand, embrace these dependencies by constructing a sequential classification process.

Essentially, a classifier chain involves transforming the output space into an ordered sequence of tasks. The prediction of each task is conditioned not only on the input but also on the predictions of all preceding tasks. This sequential process captures the conditional probabilities that standard methods overlook, often leading to considerable improvements in accuracy and, importantly, more meaningful interpretations of complex relationships between labels.

To clarify, imagine you're classifying user reviews, and labels include sentiments like "positive," "negative," and specific product aspects like "usability," "performance," and "pricing." It's conceivable that if a review is highly negative, the "pricing" aspect might be mentioned more critically. A standard transformer with independent output layers doesn’t explicitly learn this dependency. However, a classifier chain can, by first identifying the overall sentiment and then using that prediction as an input feature to classify product aspects.

Now, implementing this isn't as complicated as it may sound, but it requires careful planning and some code finesse. Let's look at a practical example using `transformers` and `sklearn` in python, a pretty common setup I’ve encountered:

```python
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Load a pre-trained transformer model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

# Simulate input data for demonstration purposes
texts = [
    "This product is fantastic and performs flawlessly.",
    "It's too expensive and the user interface is awful.",
    "The performance is adequate, but the price is not justified."
]

labels = [
    ['positive', 'performance'],
    ['negative', 'usability', 'pricing'],
    ['performance', 'pricing']
]

#Binarize labels
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(labels)

def get_transformer_embeddings(texts, tokenizer, transformer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = transformer(**inputs).last_hidden_state[:, 0, :]
    return outputs.numpy()

embeddings = get_transformer_embeddings(texts, tokenizer, transformer)


# Define and train the classifier chain
chain_base = LogisticRegression(solver='liblinear', random_state=42)
chain = ClassifierChain(chain_base, order='random', random_state=42)
chain.fit(embeddings, binary_labels)

# Prediction phase
test_text = ["This is a great product, reasonably priced and super effective!"]
test_embeddings = get_transformer_embeddings(test_text, tokenizer, transformer)
predicted_labels_binary = chain.predict(test_embeddings)
predicted_labels = mlb.inverse_transform(predicted_labels_binary)

print(f"Predicted labels for: {test_text[0]}: {predicted_labels}")
```
In this example, we’re using a basic `LogisticRegression` as our chain classifier, but you could easily substitute that for something like a more sophisticated SVM or even a lightweight neural network. The core point is the use of `ClassifierChain` from sklearn, that handles the sequencing and feature augmentation (through previous predictions) for us.

The "order='random'" parameter is crucial. It defines the sequence of classification tasks, and while random ordering can be good for initial exploration, it's often optimal to carefully choose the ordering based on your understanding of label dependencies. This is a key tuning parameter in any classifier chain setup.

Another approach involves a more explicit chaining within a neural network structure, particularly useful when the complexity justifies the overhead. Consider this illustration, again within the `torch` environment:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Load a pre-trained transformer model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)
num_labels = 5 #Number of distinct labels in our dataset

# Simulate input data for demonstration purposes
texts = [
    "This product is fantastic and performs flawlessly.",
    "It's too expensive and the user interface is awful.",
    "The performance is adequate, but the price is not justified."
]

labels = [
    ['positive', 'performance'],
    ['negative', 'usability', 'pricing'],
    ['performance', 'pricing']
]

#Binarize labels
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(labels)


def get_transformer_embeddings(texts, tokenizer, transformer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = transformer(**inputs).last_hidden_state[:, 0, :]
    return outputs

class ClassifierChainModel(nn.Module):
    def __init__(self, transformer_embedding_dim, num_labels, hidden_size=128):
        super(ClassifierChainModel, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_embedding_dim = transformer_embedding_dim
        self.num_labels = num_labels

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_embedding_dim + i, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
                ) for i in range(num_labels)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        all_outputs = []
        concatenated = embeddings
        for i in range(self.num_labels):
          output = self.sigmoid(self.classifiers[i](concatenated))
          all_outputs.append(output)
          concatenated = torch.cat((concatenated,output),axis=1)
        return torch.cat(all_outputs, axis=1)


#Initialize and train the model
embeddings = get_transformer_embeddings(texts, tokenizer, transformer)
embedding_size = embeddings.shape[1]
chain_model = ClassifierChainModel(embedding_size, num_labels)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(chain_model.parameters(), lr=0.001)

epochs = 100
binary_labels_tensor = torch.tensor(binary_labels,dtype=torch.float32)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = chain_model(embeddings)
    loss = loss_function(outputs, binary_labels_tensor)
    loss.backward()
    optimizer.step()

#Prediction example
test_text = ["This is a great product, reasonably priced and super effective!"]
test_embeddings = get_transformer_embeddings(test_text, tokenizer, transformer)
predicted_probs = chain_model(test_embeddings)
predicted_labels_binary = (predicted_probs > 0.5).int().numpy()
predicted_labels = mlb.inverse_transform(predicted_labels_binary)


print(f"Predicted labels for: {test_text[0]}: {predicted_labels}")

```
In this approach, each classifier layer's input dynamically incorporates the outputs of its predecessors, resulting in a more tightly coupled and expressive model. The advantage is that you can more finely tune not only the classifier but also its interaction with the transformer’s latent representation.

One can also explore a slightly modified version, where the output of the transformer is not passed through a sequence of classifiers but each classifier receives the transformer output concatenated with the prior classifiers output. This structure more closely resembles the initial description of classifier chains. Here is a simple illustration, which uses a `Linear` layer, rather than the full `nn.Sequential` used previously:
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Load a pre-trained transformer model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)
num_labels = 5 #Number of distinct labels in our dataset

# Simulate input data for demonstration purposes
texts = [
    "This product is fantastic and performs flawlessly.",
    "It's too expensive and the user interface is awful.",
    "The performance is adequate, but the price is not justified."
]

labels = [
    ['positive', 'performance'],
    ['negative', 'usability', 'pricing'],
    ['performance', 'pricing']
]

#Binarize labels
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(labels)


def get_transformer_embeddings(texts, tokenizer, transformer):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = transformer(**inputs).last_hidden_state[:, 0, :]
    return outputs

class ClassifierChainModel(nn.Module):
    def __init__(self, transformer_embedding_dim, num_labels, hidden_size=128):
        super(ClassifierChainModel, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_embedding_dim = transformer_embedding_dim
        self.num_labels = num_labels

        self.classifiers = nn.ModuleList([
                nn.Linear(transformer_embedding_dim + i, 1)
            for i in range(num_labels)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings):
        all_outputs = []
        concatenated = embeddings
        for i in range(self.num_labels):
          output = self.sigmoid(self.classifiers[i](concatenated))
          all_outputs.append(output)
          concatenated = torch.cat((concatenated,output),axis=1)
        return torch.cat(all_outputs, axis=1)

#Initialize and train the model
embeddings = get_transformer_embeddings(texts, tokenizer, transformer)
embedding_size = embeddings.shape[1]
chain_model = ClassifierChainModel(embedding_size, num_labels)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(chain_model.parameters(), lr=0.001)

epochs = 100
binary_labels_tensor = torch.tensor(binary_labels,dtype=torch.float32)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = chain_model(embeddings)
    loss = loss_function(outputs, binary_labels_tensor)
    loss.backward()
    optimizer.step()

#Prediction example
test_text = ["This is a great product, reasonably priced and super effective!"]
test_embeddings = get_transformer_embeddings(test_text, tokenizer, transformer)
predicted_probs = chain_model(test_embeddings)
predicted_labels_binary = (predicted_probs > 0.5).int().numpy()
predicted_labels = mlb.inverse_transform(predicted_labels_binary)


print(f"Predicted labels for: {test_text[0]}: {predicted_labels}")
```
For a deeper understanding, I would suggest delving into 'Multi-label Classification: An Overview' by Tsoumakas et al., which provides a solid theoretical background, and of course, the scikit-learn documentation itself for a very practical perspective, specifically its section on classifier chains. For neural network based implementations, you will find that papers like 'Attention is All You Need' which describes the core of transformers, can help contextualize better how to combine the two. Also, reviewing papers discussing multi-label architectures that aren't strictly based on classifier chains is useful to see the contrast.

To be clear, classifier chains aren’t a panacea. They introduce a sequential dependency that can amplify errors if not handled carefully, particularly early in the chain. Their efficacy is heavily dependent on the nature of the label dependencies, the selected classification algorithm, and careful design of the label ordering. However, in many real-world scenarios, specifically those where you deal with complex interrelated label structures, it can be a valuable tool in enhancing the performance and interpretability of transformer-based NLP models. I hope this provides some useful insights and a practical starting point for you.
