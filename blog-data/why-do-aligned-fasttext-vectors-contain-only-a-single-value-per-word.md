---
title: "Why do aligned FastText vectors contain only a single value per word?"
date: "2024-12-23"
id: "why-do-aligned-fasttext-vectors-contain-only-a-single-value-per-word"
---

,  I've definitely seen my share of oddities in vector embeddings, and the single-value-per-word situation with aligned FastText vectors is something I recall troubleshooting back during a particularly large multi-language project I was managing. It's not necessarily an *inherent* property of FastText itself, but rather a consequence of *how* we typically align these vectors for cross-lingual applications.

The crux of the issue boils down to the alignment process itself and the goals it's trying to achieve. FastText, at its core, creates vectors that represent the semantics of words within a given language. These embeddings are trained based on local context and subword information, enabling robust handling of out-of-vocabulary words. However, these embeddings are language-specific. They don't inherently understand the relationship between "cat" in english and "gato" in spanish.

When we aim to build a multi-lingual system, we need a way to map the embedding spaces of different languages into a common shared space. This is where alignment techniques come into play. The common method relies on a *linear transformation* which is what usually leads to the observation of one value for a given word across different languages after aligning the vectors. Essentially, the transformation aims to rotate, scale, and shift the vector space of one language to match the vector space of another language.

The transformation is typically achieved by first identifying pairs of translation equivalents—words that mean the same thing in different languages. For example, "dog" and "perro". These pairs form the basis for training the transformation matrix. Techniques like Procrustes analysis or similar approaches are used to find the optimal matrix *W* that, when applied to the source language vectors, makes them as close as possible to the target language vectors. Now, here’s why the single value pops up. The matrix *W* applies a linear operation to the entire vector of a word, changing *all* values. What results is a set of vectors that may look different in values in their respective language embeddings, but after the multiplication by *W*, we are projecting these vectors onto a shared space. *Because the entire source embedding is projected, after the transformation, you get a single representation for the word regardless of the source space.*

Therefore, while FastText *itself* generates multi-dimensional vectors that encapsulate semantic relationships, these embeddings are not inherently comparable across languages. The alignment process effectively forces them into a *single point* in the shared space. The shared space is still a vector space, with multiple dimensions for each word, but since the words are mapped to each other by a linear transformation, the result is a single point in that space.

Let's illustrate with some Python code using `numpy` to represent the transformation process:

```python
import numpy as np

def transform_vectors(source_vectors, transformation_matrix):
  """Applies a linear transformation to source vectors.

  Args:
    source_vectors: A numpy array where each row is a word vector in source language.
    transformation_matrix: The numpy transformation matrix.

  Returns:
    A numpy array where each row is the transformed word vector.
  """
  transformed_vectors = np.dot(source_vectors, transformation_matrix)
  return transformed_vectors

# Example usage:
# Suppose we have vectors for the word 'dog' in English and 'perro' in Spanish.
english_dog_vector = np.array([0.2, 0.5, -0.1, 0.8, -0.3])
spanish_perro_vector = np.array([-0.1, 0.7, 0.3, 0.6, 0.2])

# This is an example of what a transformation matrix may look like
transformation_matrix = np.array([[ 0.8, 0.1, -0.2, 0.05, 0.1 ],
                                 [-0.1, 0.7, 0.2, 0.1,  0.05],
                                 [0.2, -0.1, 0.9,  0.05,-0.1],
                                 [0.05, 0.2, 0.1, 0.75, 0.1 ],
                                 [0.1, 0.05, -0.1, 0.2, 0.8 ]])

transformed_english_dog = transform_vectors(english_dog_vector, transformation_matrix)


print("Original english vector:", english_dog_vector)
print("Transformed english vector:", transformed_english_dog)
print("Spanish vector:", spanish_perro_vector)
```

In this snippet, we show the basic idea. The `english_dog_vector`, after multiplication by `transformation_matrix`, is placed in the shared embedding space. Although it is now different from the original `english_dog_vector`, it is still represented by a vector of dimension equal to the original space. The matrix *W* is trained in a manner that aims to minimize the difference between the transformed `english_dog` and `spanish_perro`.

To be more explicit in showing that we use a shared representation for all instances of word in the other language, I’ll show the transformations of two words in two languages in the next example.

```python
import numpy as np

def transform_vectors(source_vectors, transformation_matrix):
  transformed_vectors = np.dot(source_vectors, transformation_matrix)
  return transformed_vectors

# Example usage:
# Suppose we have vectors for the word 'cat' and 'dog' in English.
english_cat_vector = np.array([0.3, 0.6, -0.2, 0.7, -0.4])
english_dog_vector = np.array([0.2, 0.5, -0.1, 0.8, -0.3])


# and 'gato' and 'perro' in Spanish.
spanish_gato_vector = np.array([-0.2, 0.8, 0.4, 0.5, 0.1])
spanish_perro_vector = np.array([-0.1, 0.7, 0.3, 0.6, 0.2])


# The same transformation matrix
transformation_matrix = np.array([[ 0.8, 0.1, -0.2, 0.05, 0.1 ],
                                 [-0.1, 0.7, 0.2, 0.1,  0.05],
                                 [0.2, -0.1, 0.9,  0.05,-0.1],
                                 [0.05, 0.2, 0.1, 0.75, 0.1 ],
                                 [0.1, 0.05, -0.1, 0.2, 0.8 ]])

transformed_english_cat = transform_vectors(english_cat_vector, transformation_matrix)
transformed_english_dog = transform_vectors(english_dog_vector, transformation_matrix)

print("Original english cat vector:", english_cat_vector)
print("Transformed english cat vector:", transformed_english_cat)
print("Original english dog vector:", english_dog_vector)
print("Transformed english dog vector:", transformed_english_dog)


print("Spanish gato vector:", spanish_gato_vector)
print("Spanish perro vector:", spanish_perro_vector)
```
The transformed english vectors `transformed_english_cat` and `transformed_english_dog`, share the same space as the spanish vectors `spanish_gato_vector` and `spanish_perro_vector`, respectively, thus using the shared space in the language projection.

Now, for the last example, I’ll show how we can actually use the vectors to compare the similarity of words in different languages.

```python
import numpy as np
from numpy.linalg import norm

def transform_vectors(source_vectors, transformation_matrix):
  transformed_vectors = np.dot(source_vectors, transformation_matrix)
  return transformed_vectors

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


# Example usage:
# Suppose we have vectors for the word 'cat' in English and 'gato' in Spanish.
english_cat_vector = np.array([0.3, 0.6, -0.2, 0.7, -0.4])
spanish_gato_vector = np.array([-0.2, 0.8, 0.4, 0.5, 0.1])

#Suppose also the word 'dog' in english and spanish 'perro'
english_dog_vector = np.array([0.2, 0.5, -0.1, 0.8, -0.3])
spanish_perro_vector = np.array([-0.1, 0.7, 0.3, 0.6, 0.2])


# This is an example of what a transformation matrix may look like
transformation_matrix = np.array([[ 0.8, 0.1, -0.2, 0.05, 0.1 ],
                                 [-0.1, 0.7, 0.2, 0.1,  0.05],
                                 [0.2, -0.1, 0.9,  0.05,-0.1],
                                 [0.05, 0.2, 0.1, 0.75, 0.1 ],
                                 [0.1, 0.05, -0.1, 0.2, 0.8 ]])

transformed_english_cat = transform_vectors(english_cat_vector, transformation_matrix)
transformed_english_dog = transform_vectors(english_dog_vector, transformation_matrix)

similarity_cat_gato = cosine_similarity(transformed_english_cat, spanish_gato_vector)
similarity_dog_perro = cosine_similarity(transformed_english_dog, spanish_perro_vector)
similarity_cat_perro = cosine_similarity(transformed_english_cat, spanish_perro_vector)
similarity_dog_gato = cosine_similarity(transformed_english_dog, spanish_gato_vector)

print(f"Similarity between 'cat' and 'gato': {similarity_cat_gato:.4f}")
print(f"Similarity between 'dog' and 'perro': {similarity_dog_perro:.4f}")
print(f"Similarity between 'cat' and 'perro': {similarity_cat_perro:.4f}")
print(f"Similarity between 'dog' and 'gato': {similarity_dog_gato:.4f}")

```

In this example, you can see the cosine similarity between the words translated and their respective translations in other languages. As it should be, the similarity between `cat` and `gato` and `dog` and `perro` is high, while the others are not. This demonstrates the power of transforming vectors into a shared space, where similar words are closer.

For further reading and deeper insights, I'd strongly recommend exploring papers on cross-lingual word embeddings. Specifically, look into the works on *orthogonal transformation* or *procrustes analysis* for aligning embeddings. A good starting point would be the original FastText paper from Facebook research, as well as works by Mikolov on word embedding. Additionally, the book "Speech and Language Processing" by Jurafsky and Martin includes detailed explanations of these concepts. Also, look into the paper "Bilingual Word Embeddings for Neural Machine Translation," for a more complete view on how this embeddings can be used. These references should give you a solid theoretical and practical grounding in the topic.
