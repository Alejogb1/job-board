---
title: "How can I use Huggingface Transformers FAISS index scores?"
date: "2024-12-23"
id: "how-can-i-use-huggingface-transformers-faiss-index-scores"
---

Okay, let's talk about using FAISS index scores with Hugging Face Transformers. This is a topic I've spent a fair amount of time exploring, particularly when building a semantic search system for a large document corpus back at [fictional company name] a few years ago. It's not simply about retrieving the nearest neighbors, but understanding the *meaning* of those scores in the context of your specific application, which can often be more intricate than it seems at first glance.

At the heart of this lies the fact that FAISS, the Facebook AI Similarity Search library, doesn't inherently deal in probabilities or confidences directly comparable across different queries or datasets. The scores it returns are essentially distance metrics, reflective of how far apart (in embedding space) a query vector is from indexed vectors. The lower the score, generally, the closer the vectors are. However, these scores are highly dependent on the embedding space created by your transformer model and the indexing parameters used within FAISS.

My first real experience with this was less than ideal. I had assumed a direct linear relationship between score and relevance. I quickly discovered that this assumption led to inconsistent results. A score of, say, '0.5' in one region of the embedding space could mean a far stronger semantic connection than '0.5' in another. This variability stems from the non-uniform distribution of embeddings in the high-dimensional space where these vectors reside. We were essentially seeing cluster effects; high-density regions would produce lower scores on average, regardless of the semantic match strength in a global sense.

So, what do we do? First, let's acknowledge there's rarely a 'one-size-fits-all' approach. We need to calibrate our interpretation of scores based on the context and goals of our application. I’ve found a few general approaches quite useful:

**1. Relative Ranking Within a Query:**

The most straightforward and often most valuable interpretation of FAISS scores is their relative rank for a *specific* query. Forget any absolute score thresholds initially. The lowest scoring result for a given query should be considered the *most relevant* result in the context of that single query. This provides a relative ordering of results, which often aligns quite well with human intuition. We often used this as our default approach, then augmented based on other heuristics.

Here's a basic python snippet to illustrate that:

```python
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

def embed_and_index(texts, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for text in texts:
      inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
      outputs = model(**inputs)
      embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()) # simple mean pooling
    embeddings = np.vstack(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)  # for illustrative purposes, can use faster indexes
    index.add(embeddings)
    return index, embeddings, tokenizer

def search(query, index, tokenizer, embeddings, k=5, model_name='bert-base-uncased'):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    model = AutoModel.from_pretrained(model_name)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    D, I = index.search(query_embedding, k) #D stores distances, I stores indices
    return D[0], I[0]

if __name__ == "__main__":
    texts = [
        "The cat sat on the mat.",
        "The dog chased the ball.",
        "The sun is shining brightly.",
        "A fluffy cat naps.",
        "The dog is running fast."
    ]
    index, embeddings, tokenizer = embed_and_index(texts)
    query = "A feline relaxes."
    distances, indices = search(query, index, tokenizer, embeddings)
    for i, (distance, index_val) in enumerate(zip(distances, indices)):
      print(f"Rank {i+1}: Text '{texts[index_val]}' (distance: {distance})")
```

This snippet shows how to retrieve ranked texts based on the similarity score returned by the FAISS index. Notice that the distances don’t have an obvious absolute interpretation, but the relative ranking is clear.

**2. Normalization and Transformation of Scores:**

To make scores more interpretable, we need to consider normalization. Since distances are positive, we can transform them into similarity scores by simply using `1/(1+distance)` or some similar function. Even better, we can use statistical normalization. I spent a significant amount of time exploring min-max scaling and z-score normalization, particularly after our original approach had difficulties generalising to new datasets.

The following code example demonstrates this:

```python
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def embed_and_index(texts, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for text in texts:
      inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
      outputs = model(**inputs)
      embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    embeddings = np.vstack(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings, tokenizer

def search(query, index, tokenizer, embeddings, k=5, model_name='bert-base-uncased', norm_type='minmax'):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    model = AutoModel.from_pretrained(model_name)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    D, I = index.search(query_embedding, k)
    if norm_type == 'minmax':
      scaler = MinMaxScaler()
      distances_scaled = scaler.fit_transform(D[0].reshape(-1, 1)).flatten()
      similarities = 1-distances_scaled
    elif norm_type == 'zscore':
      scaler = StandardScaler()
      distances_scaled = scaler.fit_transform(D[0].reshape(-1, 1)).flatten()
      similarities = 1/(1+distances_scaled)
    else: #no normalization
       similarities = 1/(1+D[0])
    return similarities, I[0]

if __name__ == "__main__":
    texts = [
        "The cat sat on the mat.",
        "The dog chased the ball.",
        "The sun is shining brightly.",
        "A fluffy cat naps.",
        "The dog is running fast."
    ]
    index, embeddings, tokenizer = embed_and_index(texts)
    query = "A feline relaxes."
    similarities, indices = search(query, index, tokenizer, embeddings, norm_type='minmax') #try norm_type='zscore'
    for i, (similarity, index_val) in enumerate(zip(similarities, indices)):
      print(f"Rank {i+1}: Text '{texts[index_val]}' (similarity: {similarity:.3f})")

```

Here, you see normalization to bound the scores, making them more meaningful. Experiment with different methods for normalization, and keep your final application in mind as you evaluate results.

**3. Empirical Calibration with Ground Truth:**

Finally, and perhaps most importantly, for applications where precise relevance scores matter, you'll need to establish an empirical relationship. In my experience, there is no shortcut, this approach requires you to generate a labelled dataset where you know the actual relevance scores for certain query-document pairs. Then, train a model, potentially as simple as a linear regression, to map your FAISS scores to the actual relevance score. This is essentially a supervised learning problem. I frequently used this when evaluating various FAISS indices to see what worked best for our use case.

Here's a simplified example:

```python
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def embed_and_index(texts, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for text in texts:
      inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
      outputs = model(**inputs)
      embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    embeddings = np.vstack(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings, tokenizer

def search_and_score(query, index, tokenizer, embeddings, model, k=5, model_name='bert-base-uncased'):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    model = AutoModel.from_pretrained(model_name)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    D, I = index.search(query_embedding, k)
    predicted_scores = model.predict(D.T)
    return predicted_scores[0], I[0]

if __name__ == "__main__":
    texts = [
        "The cat sat on the mat.",
        "The dog chased the ball.",
        "The sun is shining brightly.",
        "A fluffy cat naps.",
        "The dog is running fast."
    ]
    index, embeddings, tokenizer = embed_and_index(texts)

    #Generate dummy labeled data
    queries = ["A feline relaxes.", "A canine plays"]
    actual_scores = np.array([[0.9, 0.2, 0.05, 0.8, 0.1], [0.1, 0.85, 0.1, 0.2, 0.9] ]) # dummy scores
    distances = []
    for query in queries:
      inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
      model = AutoModel.from_pretrained('bert-base-uncased')
      query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
      D, I = index.search(query_embedding, 5)
      distances.append(D[0])

    distances = np.array(distances)
    X = distances.T
    y = actual_scores.T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    query = "A feline relaxes."
    predicted_scores, indices = search_and_score(query, index, tokenizer, embeddings, model)
    for i, (score, index_val) in enumerate(zip(predicted_scores, indices)):
      print(f"Rank {i+1}: Text '{texts[index_val]}' (predicted score: {score:.3f})")
```

This shows a simplified example where we train a linear regression on FAISS scores to map them to more human interpretable relevance scores. In practice, you might have a much larger training set, a more complex model, and features derived from not only the FAISS scores but potentially other factors as well.

For a more rigorous understanding of FAISS, I highly recommend looking at the original FAISS paper by Johnson, Douze, and Jégou (2017) on efficient and robust approximate nearest neighbor search. As for general embedding spaces and transformers, "Attention is All You Need" by Vaswani et al. (2017) would be the seminal work. Further, for a deep dive into vector space models and retrieval, "Introduction to Information Retrieval" by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze is an excellent choice.

In short, understanding FAISS scores requires a nuanced approach. Begin with relative ranking, explore score normalization and transformations, and calibrate with empirical data if precision is essential for your use case. It is rarely a straightforward translation; rather, it is a process of understanding the scores and aligning them with your specific application's needs.
