---
title: "How do I use bert-large-uncased in HuggingFace for long text classification?"
date: "2024-12-23"
id: "how-do-i-use-bert-large-uncased-in-huggingface-for-long-text-classification"
---

Okay, let's tackle this. I've definitely been in the trenches with large language models and long text myself, and it’s a problem that pops up quite frequently. The crux of it is that `bert-large-uncased`, like most transformer models, has a fixed input length. If you feed it text exceeding that limit, usually around 512 tokens for BERT, it just truncates, discarding vital context. So, let’s discuss how we work around that limitation effectively for classification tasks.

When I first encountered this issue, it was with a substantial corpus of legal documents. Naively attempting to feed them directly into BERT resulted in… well, let’s just say the performance was far from ideal. I had to rethink the approach, and that's where strategies like chunking, and sliding windows came into play.

The primary challenge is that these models are designed to operate on sequences of a limited size because of the inherent computational cost of the attention mechanism. We can't just magically increase the input size without a significant hit to memory and processing speed. The core strategy involves processing the long text in manageable pieces and then aggregating the results to form a cohesive classification. There isn’t one ‘silver bullet’, but rather a collection of methods that we can use depending on the specifics of the task.

First, let’s consider **chunking**. This is the simplest approach; we divide the long text into smaller, fixed-size chunks that fit within BERT's input limit. Each chunk is passed through BERT independently, and we obtain a set of representations for them. We then need a method to combine these chunk embeddings to make a classification. This could be averaging, taking the maximum value across dimensions, or more complex techniques. This method is less computationally expensive than overlapping approaches, but it can break apart important context across chunk boundaries. The main idea is to encode the entire document in smaller sections, then use the encodings.

Here's an example of basic chunking:

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

def chunk_text(text, max_length=512, stride=0):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length - stride):
        chunk = tokens[i:i + max_length]
        chunks.append(chunk)
    return [" ".join(chunk) for chunk in chunks]

def get_chunk_embeddings(text_chunks):
    all_chunk_embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        all_chunk_embeddings.append(embeddings)
    return all_chunk_embeddings

text = "This is an example of a very long text that we want to classify. It contains multiple sentences and should be broken into chunks to avoid going over BERT's input limit. We need to analyze all of it to make the best classification decision." * 10
text_chunks = chunk_text(text, max_length=512)
chunk_embeddings = get_chunk_embeddings(text_chunks)
combined_embedding = np.mean(np.array(chunk_embeddings), axis=0) # Simplest combination, you might do something else

print("Combined embedding shape:", combined_embedding.shape)
```

In this first code snippet, we're using the most straightforward chunking approach, without any overlap. The `mean` operation in `combined_embedding` provides a rudimentary aggregation. As you might imagine, depending on your task and data, this may be far from sufficient. The performance of this first example is acceptable for basic text analysis tasks where context across chunks is not vital. However, it’s a start.

Next, we can implement a **sliding window** technique. This method is similar to chunking but introduces overlap between chunks. The overlap ensures that no context is lost at the edges of the chunks. This adds complexity, but often enhances accuracy. Each window of text is encoded by the model, and then we need a means to combine the multiple window representations. The trade-off here is increased computation for hopefully better results compared to simple chunking.

Let's look at a code example:

```python
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertModel.from_pretrained('bert-large-uncased')

def sliding_window(text, max_length=512, stride=256):
    tokens = tokenizer.tokenize(text)
    windows = []
    for i in range(0, len(tokens), max_length - stride):
        window = tokens[i:min(i+max_length, len(tokens))]
        windows.append(window)
        if i + max_length >= len(tokens):
           break

    return [" ".join(window) for window in windows]

def get_window_embeddings(text_windows):
    all_window_embeddings = []
    for window in text_windows:
         inputs = tokenizer(window, return_tensors='pt', padding=True, truncation=True, max_length=512)
         with torch.no_grad():
            outputs = model(**inputs)
         embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
         all_window_embeddings.append(embeddings)
    return all_window_embeddings

text = "This is an example of a very long text that we want to classify. It contains multiple sentences and should be broken into chunks to avoid going over BERT's input limit. We need to analyze all of it to make the best classification decision." * 10
text_windows = sliding_window(text, max_length=512, stride=256)
window_embeddings = get_window_embeddings(text_windows)
combined_embedding = np.mean(np.array(window_embeddings), axis=0) # Can be something else

print("Combined embedding shape:", combined_embedding.shape)
```
Here, the `stride` variable introduces an overlap between the text windows, and we combine window embeddings, similar to what was shown in the first code snippet. It is important to note that you can choose different aggregations. For tasks with strong dependencies between subsequent chunks, this tends to give better classification outcomes. This improved the performance in many past projects.

Finally, there's also a strategy based on **hierarchical document embeddings**. In this strategy, you might use something like the sentence-transformer approach to generate sentence embeddings first, then group these into paragraph representations and ultimately document embeddings. This strategy is more complex to implement but could preserve fine-grained semantic structure for extremely long text.

While I won't provide the code for the full hierarchical embedding here since it's quite a bit more involved, I will show how you could use Sentence-transformers to get sentence embeddings which could then be used in your strategy:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

sentences = [
    "This is the first sentence.",
    "This is the second sentence. It has some additional information.",
    "And finally the third sentence concludes this short paragraph."
]

sentence_embeddings = model.encode(sentences)
combined_embedding = np.mean(sentence_embeddings, axis=0) # you can do more complex aggregations


print("Sentence embedding shape:", sentence_embeddings.shape)
print("Combined embedding shape:", combined_embedding.shape)
```
This shows you how you can easily generate sentence embeddings and then perform some form of aggregation. This would form a base layer for further hierarchal encoding.

Choosing the correct method depends highly on the problem and data. Remember that these are not mutually exclusive, and you could potentially combine different strategies based on the data's structure. For example, you might use hierarchical embeddings for very large documents and sliding windows within each sub-document or section.

For further learning, I'd highly recommend going through the original BERT paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by Devlin et al. There is also a great chapter in "Speech and Language Processing" by Daniel Jurafsky and James H. Martin that covers these concepts in more depth. Also, reading research papers on hierarchical classification and long-document processing specifically is highly beneficial. Finally, a very practically useful book, "Natural Language Processing with Transformers" by Tunstall, von Werra, and Wolf, is very useful for implementations. Understanding how to effectively utilize the features of these models is critical for any practitioner.

In conclusion, handling long texts with models like `bert-large-uncased` requires careful consideration and strategic approaches. Chunking, sliding windows, and hierarchical encoding offer solid starting points. Experiment with these options, fine-tune based on your specific task, and make sure you evaluate effectively on held-out data. Good luck, and feel free to ask for more help as you run into specific roadblocks.
