---
title: "How to Build Effective Retrieval Augmented Generation Systems"
date: "2024-11-16"
id: "how-to-build-effective-retrieval-augmented-generation-systems"
---

hey dude so i just watched this killer talk about retrieval augmented generation rag and it's totally blowing my mind  it's all about making ai way smarter by giving it a better memory  think of it like giving your brain a massive upgrade so it can actually remember and use everything it's learned instead of just kinda forgetting stuff after a few seconds

the whole point of the talk was to show how basic vector search isn't enough for serious rag apps  the guy anton from chroma was like "yeah vector search is cool and all but it's like using a flip phone in the age of smartphones"  he showed this basic rag loop diagram – you know the one with the corpus of docs embeddings vector store nearest neighbors llm  it's the standard stuff  but he was saying that's just the tip of the iceberg

one thing that really stood out was how he stressed the need for human feedback in the loop  he literally said "without human feedback it isn't possible to adapt the data the embeddings model itself to the specific task to the model and to the user"  it's like training a dog you gotta give it treats and corrections to get it to do what you want  otherwise your ai is just gonna be rambling nonsense


another key idea was about agents and world models  he was talking about how agents need to learn and adapt from their interactions with the world – and to do that their memory needs to be dynamic and constantly updating  this isn't your grandpa's static database  we're talking about a living breathing memory for the ai  he even showed this animation from the voyager minecraft paper  showing how an agent learned to play minecraft by building up its knowledge and skills and remembering them  it was crazy


then he got into the nitty-gritty of the challenges with retrieval  it's not just about finding relevant info it's about *avoiding* irrelevant info because those distractors totally screw up the ai's performance  he said "distractors in the model context cause the performance of the entire AI based application to fall off a cliff"  so finding relevant info AND avoiding irrelevant info is key to success

he broke down several challenges:

1  **picking the right embedding model:** he said  "the only way to find out which is best for your data set is to have a a effective way to figure that out"  there's no one-size-fits-all  you gotta experiment

2  **data chunking:** this is how you break down your data for the llm to process  he mentioned nltk langchain and llama index as tools but also talked about using model perplexity to detect semantic boundaries  a really cool idea  imagine using the model's own uncertainty as a guide to splitting up your data

3  **determining result relevance:**  this is huge  how do you tell if the results the ai pulled up are actually useful this isn't just about "closeness" in embedding space  he gave the example of querying about fish when your data is all about birds   the nearest neighbors might be totally irrelevant  to solve this  he talked about human feedback auxiliary reranking models and conditional relevancy signals a super complex challenge  he even hinted at using lightweight llms to assess relevance  that's some serious next-level stuff

here's a code snippet illustrating a simple embedding and similarity search  imagine this is part of a larger rag system:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# sample embeddings
query_embedding = np.array([0.2, 0.5, 0.1, 0.8])
doc_embeddings = np.array([
    [0.1, 0.6, 0.2, 0.7],
    [0.3, 0.4, 0.0, 0.9],
    [0.7, 0.2, 0.5, 0.1]
])

# calculate cosine similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)

# find the most similar documents
most_similar_indices = np.argsort(similarities[0])[::-1]

print(f"Most similar documents: {most_similar_indices}")

```

this is a basic example but you can expand on this by using a proper vector database like chroma or pinecone for much larger datasets

another code snippet about chunking using a simple sliding window approach

```python
def chunk_text(text, window_size, overlap):
  chunks = []
  for i in range(0, len(text), window_size - overlap):
      chunk = text[i:i + window_size]
      chunks.append(chunk)
  return chunks

text = "this is a long string that needs to be chunked into smaller pieces"
window_size = 10
overlap = 5
chunks = chunk_text(text, window_size, overlap)
print(chunks)
```

this function breaks the text into overlapping chunks making sure no info is lost and helping to maintain context


and here's one on using a simple linear transformation to project embeddings from one model space to another a truly mind bending concept that he mentioned

```python
import numpy as np

# sample embeddings from model A
embeddings_a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# sample embeddings from model B (these would be obtained from a different model)
embeddings_b = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])

# calculate the linear transformation (this requires a training dataset!)
# here we are simplifying  a real world implementation would be significantly more complex and require fitting a model
transformation_matrix = np.array([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1], [0.4, 0.0, 0.6]])

# apply the transformation
transformed_embeddings = np.dot(embeddings_a, transformation_matrix)


print("Original Embeddings Model A:\n",embeddings_a)
print("\nTransformed Embeddings:\n", transformed_embeddings)
print("\nOriginal Embeddings Model B:\n",embeddings_b)

```

keep in mind this is a simplified illustration  finding the correct transformation matrix is often a non-trivial task requiring sophisticated machine learning techniques

in the end anton talked about what chroma is building  a scalable cluster version of their vector database a cloud service and support for multimodal data  it's all about making rag easier to use and more powerful  he ended with "chroma wants to do everything in the data layer for you so that just like a modern dbms just like you use postr in a web application everything in the data layer for as an application developer should just work"  basically they're building the plumbing so you can focus on the fun stuff

so yeah  that was the talk  totally mind blown  rag is no joke  and it's only gonna get crazier  the future of ai is here and it's all about smarter memory systems
