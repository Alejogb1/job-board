---
title: "Why can't BERT be fitted?"
date: "2025-01-30"
id: "why-cant-bert-be-fitted"
---
Large language models like BERT, despite their powerful capabilities in natural language processing, cannot be directly “fitted” in the way traditional machine learning models are. This inability stems from the architecture and pre-training methodology inherent in these transformer-based systems, which fundamentally diverge from the supervised learning paradigm of models like logistic regression or support vector machines. My experience working with transformer architectures, specifically on a project involving cross-lingual document retrieval, revealed this constraint directly.

The core issue is that BERT, and similar models, are *pre-trained* on vast amounts of unlabeled text data to learn general-purpose linguistic representations. This pre-training phase, often involving tasks like masked language modeling and next sentence prediction, yields a model with millions or even billions of parameters capable of understanding intricate contextual relationships within text. The parameters, after this pre-training, contain a wealth of knowledge about language. Consequently, we don’t fit BERT in the conventional sense of adjusting parameters during training to learn a specific task. Instead, we *fine-tune* it.

The pre-training process essentially gives BERT a strong foundation in general language understanding; it's learned to recognize patterns, grammar, and semantic relationships across a diverse corpus. Traditional fitting, as applied to other models, would involve initializing parameters randomly and then optimizing them to minimize error on the target data. If we were to attempt this with BERT, two significant problems would arise. Firstly, the model has so many parameters that training it from scratch on a task-specific dataset would be incredibly resource intensive and likely converge to a poor local optimum with little understanding of the complexities of language. Secondly, we would be discarding valuable language information accumulated through its pre-training, effectively hindering the effectiveness of the model. Fine-tuning, conversely, leverages this pre-existing knowledge by adapting it to the nuances of the downstream task, like text classification or question answering.

Fine-tuning involves adding a small, task-specific layer on top of the pre-trained BERT model and then optimizing the parameters of this new layer *and* the parameters of the BERT model itself (though at a much slower learning rate). The bulk of the pre-trained model’s knowledge remains untouched, providing the necessary general linguistic understanding. The fine-tuning process essentially directs this foundational knowledge towards the specific demands of a particular task. This process typically involves significantly less data and computational resources compared to training a transformer from scratch.

To better illustrate, consider this first code example using Python with the `transformers` library, a commonly utilized tool for working with such architectures. Assume we want to perform text classification:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.optim import AdamW

# 1. Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) # binary classification

# 2. Prepare your data (simplified example - in reality, more extensive preparation is needed)
texts = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]  # 1 for positive, 0 for negative

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(labels)

# 3. Setup the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. Training loop (simplified)
model.train()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Here, we load the pre-trained model and tokenizer directly.  We *do not* initialize the BERT model with random weights; we are loading pre-trained weights.  The optimization process through AdamW adjusts the parameters of the *entire* model, including BERT itself, although typically with a smaller learning rate than the added classifier layer. We're not 'fitting' BERT, instead we're fine-tuning it. The core BERT weights are still based on the pre-training process.

Let's consider a slightly more involved scenario using a model for question answering:

```python
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from torch.optim import AdamW

# 1. Load the pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# 2. Prepare your data
question = "What is the capital of France?"
context = "France is a country in Europe. Its capital is Paris."
inputs = tokenizer(question, context, return_tensors='pt')

# 3. Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. Training Loop (simplified): Note, in actual QA training there will be a label indicating where the answer can be found in the context
model.train()
outputs = model(**inputs, start_positions=torch.tensor([22]), end_positions=torch.tensor([26])) #Start at the 'P' in Paris, End at the end of Paris
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Again, note that the `BertForQuestionAnswering` model is also pre-trained, with additional task-specific layers. During training, gradients are computed, and the optimizer updates both these new layer’s parameters and the parameters of the pre-trained model, effectively adapting the model to the question answering task. This example makes it even clearer: the optimization is not on randomly initialized weights, but on pre-existing information contained in the pre-trained BERT.

Finally, let's look at an example where we try to fine-tune BERT for sequence tagging. This is another common downstream application, where we might want to identify named entities, for instance.

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch
from torch.optim import AdamW

# 1. Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels = 3) #Three labels in our dummy case

# 2. Example data
text = "John Smith went to London"
tokens = tokenizer(text, return_tensors='pt')
labels = torch.tensor([0, 1, 2, 0, 0]) # A sample tagging: 0 - O, 1 - B-PER, 2 - B-LOC

# 3. Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

#4. Training loop (simplified)
model.train()
outputs = model(**tokens, labels = labels)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Here, we use `BertForTokenClassification`, which also loads a pre-trained model and adds a layer for predicting labels for each input token. As before, when we train this model, the optimizer will adjust all the parameters including the pre-trained ones. This adaptation is what allows the model to solve a specific task, not from random initial weights, but by building on pre-existing language knowledge.

In summary, fitting a model implies training it from scratch on task-specific data, typically with randomly initialized weights. BERT and similar models, on the other hand, undergo pre-training which provides a significant head start on language understanding. We instead *fine-tune* by adapting the model, using a task-specific layer, and adjust the pre-existing weights slightly to the target task. This process leverages the language knowledge ingrained during pre-training. Attempts to fit BERT from scratch would be both computationally wasteful and would also discard the pre-trained knowledge necessary for the model to perform effectively.

For further exploration, I would recommend exploring resources focused on the following: Transformer architectures, particularly the original “Attention is All You Need” paper; detailed tutorials on using the `transformers` library; and explanations of the pre-training objectives for BERT (masked language modeling and next sentence prediction). Textbooks or publications specializing in deep learning for natural language processing can offer a more complete understanding of the underlying theory.
