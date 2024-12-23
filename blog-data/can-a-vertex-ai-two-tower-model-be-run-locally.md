---
title: "Can a Vertex AI two-tower model be run locally?"
date: "2024-12-23"
id: "can-a-vertex-ai-two-tower-model-be-run-locally"
---

Alright, let's tackle this. You're asking about running a Vertex AI two-tower model locally, and it's a pertinent question, especially when considering the typical cloud-centric workflow for these types of models. Now, the direct answer is: yes, it's absolutely possible, but it comes with certain caveats and requires a methodical approach, different from simply pulling down an already trained artifact and running it. I've personally wrestled with this kind of setup multiple times during the scaling phase of projects when rapid prototyping and iteration were paramount but not always conducive to constantly pushing and pulling updates through cloud pipelines.

The core challenge with a Vertex AI two-tower model isn’t the architecture itself – which is essentially two separate neural networks, often encoders, that project data into a shared embedding space – but rather the training and deployment pipelines that Vertex AI automates. When you’re training in Vertex AI, a lot of the environment configuration, data handling, and versioning are taken care of. Moving the inference part of that whole process to a local environment requires replicating parts of that infrastructure, primarily focusing on loading the trained weights of your towers and preparing your input data according to the models’ expectations.

Let's break it down step-by-step. First, you must understand that the weights themselves are not intrinsically linked to the Vertex AI platform once training is complete. These weights, which are essentially numerical parameters, can be exported as standard TensorFlow or PyTorch saved model formats, depending on what framework you used during training. This is crucial: you're not trying to relocate an entire VM; you are extracting the trained model.

Here’s the typical flow I follow when transitioning from cloud training to local inference:

1. **Exporting the Model:** After training on Vertex AI, the trained model(s) are usually stored in Google Cloud Storage (GCS). You'll need to download these model directories to your local machine. These directories will contain the model's graph definition, weights, and, importantly, the preprocessing steps. Note, your Vertex AI configuration will often include specific preprocessing layers in the model directly, such as tokenizers, or image resizing etc. Be absolutely certain you've exported all parts of the model graph, not just the weight files.

2. **Setting up the Local Environment:** You will need a Python environment that matches the environment used during training, including framework versions (TensorFlow, PyTorch, etc.), libraries, and any custom components. This is a common pitfall. It’s critical to pay attention to the version numbers, as compatibility issues can manifest as subtle, difficult to trace bugs.

3. **Loading the Model:** You use the appropriate APIs (e.g., `tf.saved_model.load` for TensorFlow) to load the saved model directories you downloaded. Here, you need to load both 'towers' or encoding models individually, as they are separate during inference and potentially have different input shapes. This is where the flexibility of a two-tower architecture, with distinct encoders, becomes apparent.

4. **Pre-processing:** Remember, your model was likely trained with specific data preprocessing steps. You have to replicate that preprocessing logic locally, using the same tokenizers, normalization routines, padding, etc. If this step isn’t consistent with training, the performance will be markedly reduced. This will usually involve examining the saved model configuration as well as the training setup, to extract the preprocessing steps you used.

5. **Inference:** Finally, you create functions that take your input data, pre-process it correctly, pass it through your tower models individually, and obtain the resulting embeddings for each tower’s output. You’ll also need to write the inference logic that then computes a similarity score based on these embeddings.

Here's a code snippet, assuming you are working with TensorFlow, to illustrate how to load two towers individually:

```python
import tensorflow as tf
import numpy as np # for dummy input data generation

def load_two_tower_models(model_dir_tower_a, model_dir_tower_b):
    """Loads two-tower models from directories."""
    tower_a = tf.saved_model.load(model_dir_tower_a)
    tower_b = tf.saved_model.load(model_dir_tower_b)
    return tower_a, tower_b

# dummy paths
model_dir_tower_a = './model_tower_a'
model_dir_tower_b = './model_tower_b'

# Assuming model saved with following structure:
#
# model_dir_tower_a/
#   saved_model.pb
#   assets/
#   variables/
#
# Same for tower_b
# (If no saved model is found in these locations, you will have an error.)

try:
    tower_a_model, tower_b_model = load_two_tower_models(model_dir_tower_a, model_dir_tower_b)

    # Generate Dummy input data of expected size
    input_a = tf.constant(np.random.rand(1, 100), dtype=tf.float32) # batch of 1, 100 features
    input_b = tf.constant(np.random.rand(1, 128, 128, 3), dtype=tf.float32)  # batch of 1 image
    embedding_a = tower_a_model(input_a)
    embedding_b = tower_b_model(input_b)

    print("Embedding A:", embedding_a)
    print("Embedding B:", embedding_b)

except Exception as e:
  print("Failed to load or run models:", e)
```

Here’s an example of preprocessing, assuming your data is text. (Note: You will need to have the tokenizer vocab file with your model weights)

```python
import tensorflow as tf
import tensorflow_text as tf_text # requires `pip install tensorflow-text`

def preprocess_text(text_data, tokenizer_dir):
    """Tokenizes input text using a saved tokenizer model."""
    tokenizer = tf_text.SentencepieceTokenizer(model=tf.io.read_file(tokenizer_dir))
    tokens = tokenizer.tokenize(text_data)
    return tokens

tokenizer_dir = './tokenizer_model.spm' # replace with your tokenizer vocab file.

# Example Usage:
text_input = tf.constant(["This is an example sentence."])
tokenized_input = preprocess_text(text_input, tokenizer_dir)

print("Tokenized Input:", tokenized_input)
```

Finally, you might want to calculate similarity between the embeddings, like cosine similarity. This would likely be done after getting embeddings from each tower. Here is how you might calculate that:

```python
import tensorflow as tf

def compute_similarity(embedding_a, embedding_b):
    """Computes the cosine similarity between two embeddings."""
    embedding_a = tf.nn.l2_normalize(embedding_a, axis=1)
    embedding_b = tf.nn.l2_normalize(embedding_b, axis=1)
    similarity = tf.matmul(embedding_a, embedding_b, transpose_b=True)
    return similarity


# Assume embedding_a and embedding_b are tensors from previous step
# Use the dummy embeddings we generated previously
similarity_score = compute_similarity(embedding_a, embedding_b)

print("Similarity Score:", similarity_score)
```

These code snippets provide basic illustrations of the concepts. In practice, you'll likely need to add batching, error handling, and potentially other custom logic based on the complexities of your model.

For further reading on the specifics of two-tower models and how they are trained and deployed with TensorFlow, I recommend looking into “TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems” by Martín Abadi et al. This paper, although a little older, provides the foundational understanding. For practical implementation techniques, the TensorFlow documentation on `tf.saved_model` is invaluable. Specifically, pay attention to documentation on how `tf.keras` integrates with this format. Lastly, "Deep Learning with Python" by François Chollet can provide a good foundation in using the Keras API for implementing such networks and preparing them for inference.

In summary, running a Vertex AI two-tower model locally is more involved than simply copying files; it's about understanding the underlying mechanisms, replicating the essential parts of the pipeline, and ensuring consistency. By following the steps outlined and paying close attention to details like correct versions, input shapes and your model's preprocessing steps, you'll be able to smoothly transition to local inference.
