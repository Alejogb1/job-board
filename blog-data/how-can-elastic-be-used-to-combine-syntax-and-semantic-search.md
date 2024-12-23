---
title: "How can Elastic be used to combine syntax and semantic search?"
date: "2024-12-23"
id: "how-can-elastic-be-used-to-combine-syntax-and-semantic-search"
---

Okay, let’s tackle this. I’ve seen my share of search implementations, and the blend of syntax and semantics is often where things get…interesting. It’s not just about keyword matching or some fancy natural language processing (nlp) bolt-on. It’s about understanding the user's *intent* in conjunction with the *structure* of their query and the data itself.

In my past life, managing the backend search infrastructure for a content platform with millions of articles, we ran into this problem head-on. Pure keyword-based searches were returning a lot of noise, and relying solely on pre-trained embeddings proved too computationally expensive and sometimes not granular enough for our specific needs. We had to devise a hybrid approach combining the strengths of both.

First, let’s break down what we mean by "syntax" and "semantics" in this context. *Syntax* is essentially the structure of the query. Think of it as the specific words used, their order, and any Boolean operators or wildcards involved. Elastic's full-text search capabilities, like its query DSL with `match` and `bool` queries, are masters of syntax. They’re excellent at finding documents containing specific keywords and at adhering to filters.

*Semantics*, however, is about meaning. It’s the intention behind the user's query. This usually involves a deeper understanding of relationships between words and concepts. For example, searching for "fast cars" should ideally return results containing terms like "high-performance vehicles," "sports cars," or even specific car models known for speed. This goes beyond literal keyword matching.

To blend these two successfully with Elastic, I found a three-pronged strategy particularly effective:

**1. Leveraging Elastic’s Full-Text Search for Syntax, but with Added Boost:**

The basic full-text search capabilities are where you start. However, we need to tune them carefully. Instead of just blindly matching keywords, we can use boosts to prioritize fields. For example, if we're searching within blog posts, a match in the title should be given a higher boost than a match in the body. Similarly, short-term, keyword-based matches are often a strong, immediate intent indicator and can be favored initially.

Here's a basic example of a boosted query:

```json
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "title": {
              "query": "quantum computing",
              "boost": 3
             }
          }
        },
        {
          "match": {
            "body": "quantum computing"
          }
        }
      ],
    "minimum_should_match": 1
    }
  }
}
```

This query will prioritize results where "quantum computing" appears in the `title`, but will still match those where it only occurs in the `body`. This boosts relevance based on syntactic location. It’s not just “does it contain the word”, but “where does it contain the word”. You can use other parameters like `fuzziness` and `slop` to accommodate minor misspellings and variations in word order. I would refer you to Elastic's official documentation concerning the match and bool query types for further exploration, particularly focusing on advanced options, along with books like "Elasticsearch in Action, Second Edition" by Radu Gheorghe et al., which goes into great detail on this aspect of the framework.

**2. Introducing Semantic Search with Text Embeddings and Approximate Nearest Neighbor Search (ANN):**

Here's where we start understanding the meaning behind words. Instead of using just plain text for matching, we can transform our document text into dense vector representations (embeddings). Libraries like sentence-transformers (which are excellent for sentence and paragraph embeddings) or even the built-in vector fields within Elastic can be used for this transformation. These embeddings capture the semantic essence of a text. A sentence like “the vehicle was speedy” will have a vector that’s more similar to “the car had high performance” than to something like “the flower is red”. Once your text is vectorized, we utilize approximate nearest neighbor (ann) search capabilities. The crucial point here is that this allows us to find documents that are *semantically similar* to a query, even if they don’t share the exact keywords. Elastic's built-in `dense_vector` field type with the `knn` query (for k-nearest neighbors) is perfectly suited for this.

Here's an example of indexing documents with dense vectors:

```json
PUT /my_index
{
  "mappings": {
    "properties": {
      "text": {
        "type": "text"
      },
      "text_vector": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

You will first need to obtain the embeddings, then insert your documents and their related text vectors.

And this is how you’d conduct a knn search:

```json
{
  "query": {
      "knn": {
        "text_vector": {
          "vector": [0.1, 0.2, ... , 0.7],
          "k": 10
        }
      }
  }
}
```

This snippet searches for the top 10 documents whose `text_vector` are most similar to the provided vector. The vector here would have to be created based on the search query using your chosen embedding method. The key advantage is that now the results aren’t limited to the literal keyword match, but the conceptual similarity. A good reference to study how to implement this would be “Natural Language Processing with Transformers” by Lewis Tunstall et al. It provides a comprehensive study of text embedding, transformer model implementation, and fine-tuning.

**3. Combining Results and Using Hybrid Scoring:**

The real magic happens when we combine both approaches. In the simplest case, we could execute a full-text query, and separately a knn query, and then simply merge and re-rank the results. This works well in many situations. However, we can achieve better results by combining scores directly in the query itself. This can be done using a `function_score` query. Here, we can assign different weights to syntax and semantic score components. You can use arbitrary calculation here, not just linear weighted sums.

Here's an example of a hybrid score:

```json
{
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "should": [
            {
              "match": {
                "title": {
                  "query": "artificial intelligence",
                  "boost": 3
                 }
              }
            },
            {
              "match": {
                "body": "artificial intelligence"
              }
            }
          ],
        "minimum_should_match": 1
        }
      },
    "functions": [
        {
          "filter": { "match_all": {} },
          "script_score": {
            "script": {
                "source": "_score + 0.5 * doc['text_vector'].knn(params.query_vector, params.k)",
                "params": {
                  "query_vector": [0.1, 0.2, ... , 0.7],
                  "k": 10
                }
              }
          }
        }
      ],
    "score_mode": "multiply"
    }
  }
}
```

This query executes the full-text search as before, but in addition, it executes the knn similarity operation and the resulting score is added to the base `_score` value (the result of the boolean query). You would want to scale these scores correctly depending on your use case, but the idea is clear: we are adding the semantic score directly to the syntactic relevance score. You could also do more complex things, like implement a completely new score using arbitrary algorithms and data retrieved using the query parameters (you are not limited to simple additions here), giving more precise control over how scores are generated. For further reference, I’d recommend reviewing the “Lucene in Action, Second Edition” book, which describes the underpinning concepts of Elastic's score generation and highlights advanced customization strategies.

**Practical Considerations and Lessons Learned:**

Implementing such a system isn’t trivial. It requires careful consideration of several factors. The embeddings model you choose will have a significant impact on the results, and I’d recommend experimenting with multiple models to find one that suits your specific content. The dimensions of the dense vector have a direct impact on the search speed, but also on the quality of semantic retrieval, so that would require experimentation too. Keep in mind vector searches are more computationally intensive than traditional text searches, so scaling and optimization should be considered. You also need to develop a plan for updating embeddings when documents change or when you need to fine-tune or adopt an updated model.

In the past, we initially underestimated the importance of testing and performance profiling, which resulted in the need for significant adjustments to the query structure and the embedding strategies. Always conduct thorough performance tests under production-like conditions to evaluate efficiency and relevance of your setup.

In summary, combining syntactic and semantic search with Elastic isn't about adopting one over the other; it's about utilizing each approach’s strengths to complement the other. By carefully tuning your full-text searches, introducing semantic embeddings and ann searches, and combining the scores, you can create a powerful search solution that provides a far more intuitive and relevant experience for your users. This is, after all, our goal as engineers.
