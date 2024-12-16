---
title: "Why do Transformer summarizer results vary with a fixed seed?"
date: "2024-12-16"
id: "why-do-transformer-summarizer-results-vary-with-a-fixed-seed"
---

Okay, let’s unpack why transformer summarization results, even with a fixed seed, can exhibit variation. It’s a question I’ve certainly grappled with more than once, particularly during a past project involving large-scale document analysis for legal review. We were initially relying heavily on the deterministic nature of a fixed seed to ensure consistent, reproducible outcomes, but the reality proved a little more nuanced. This isn't a case of simple randomness; it's the interaction of several stochastic elements within the transformer architecture itself and during the training and inference pipelines.

At the core of a transformer, and indeed many deep learning models, lies the fact that many processes aren't entirely deterministic, even when we try to control them with a fixed random seed. We might think a fixed seed for numpy, torch, or tensorflow will fully reproduce a run, but this is only partially true, and often insufficient for completely identical results in complex pipelines. What we're really seeding is the initialization of weights, and the random number generation for processes like dropout, which operates during training and *sometimes* during inference, if that’s how it’s implemented by the library and your chosen model.

One of the most significant sources of variation, particularly in summarization, stems from the stochastic nature of training. Gradient descent is inherently an iterative process, where the model’s weights are adjusted based on gradients calculated on mini-batches of data. Now, even if the initial weights are the same, and the random numbers for dropout are identical, the order in which these batches are processed during training will generally not be identical (unless you are meticulously ensuring full determinism at every point). Differences in these sequences affect how the gradients are computed and applied, causing slight variations in the final learned weights. This is precisely where seemingly identical starting points can lead to different final model states, even with the same seed.

Furthermore, the tokenization process introduces some variability. While the tokenization *algorithm* is deterministic, the exact vocabulary, which forms the numerical mappings, influences the representation and hence the training. Variations in how the vocabulary is built, especially from different training runs using slightly modified data or parameters, lead to different numerical inputs to the neural network itself.

Inference, too, has sources of subtle randomness despite the fixed seed, unless additional steps for full determinism are taken. Some transformer models use dropout *during inference* for uncertainty estimation (which might be disabled in some inference pipelines). The order of input sequences and the specific device on which the inference runs (gpu vs cpu, different gpu models) can also introduce small variations. Numerical precision issues across different hardware can also contribute to these differences - although usually minor, these can amplify in complex models during the iterative nature of inference.

Let's look at some practical coding snippets to illustrate these points.

**Example 1: Demonstrating how different training runs can result in different model outputs even with the same seed.**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_summarize(model_name, text, epochs=1, seed=42):
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    labels = tokenizer("Summarized Text", return_tensors="pt", max_length=128, truncation=True)['input_ids']

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels = labels.cuda()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    summary_ids = model.generate(inputs['input_ids'], max_length=120, num_beams=4, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# Simple example text
text = "This is a long document that needs summarizing. It has many sentences, and we want to produce a good summary. The transformer is a powerful architecture, but its stochastic nature can be surprising."

# Training with fixed seed, twice, demonstrating variability.
summary1 = train_and_summarize("t5-small", text, epochs=1, seed=42)
summary2 = train_and_summarize("t5-small", text, epochs=1, seed=42)
print("Summary 1:", summary1)
print("Summary 2:", summary2)


```

Running this snippet, while it only does a very limited training loop (for speed purposes), will often result in slightly different summaries, illustrating how the stochastic nature of the training loop can lead to variances. This difference may not be immediately obvious, but it highlights how subtle changes in weight training can cause diverse text generation output.

**Example 2: Demonstrating effects of different GPUs**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)

def summarize_with_gpu(model_name, text, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    if torch.cuda.is_available():
       inputs = {k: v.cuda() for k, v in inputs.items()}

    summary_ids = model.generate(inputs['input_ids'], max_length=120, num_beams=4, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

text = "This is the text I want to summarize for this test. It has many different aspects and points that I want covered in the summary. I am making this a few sentences to help make this example more realistic."

# Note: This may only show variation on certain machines, and if the same GPU is used
#  this may result in identical outputs. For true cross-device tests, we need access to multiple GPUs and run on them
#  for comparison which can't be done in this environment.
summary_gpu1 = summarize_with_gpu("t5-small", text, seed=42)
summary_gpu2 = summarize_with_gpu("t5-small", text, seed=42)


print("GPU1 summary: ", summary_gpu1)
print("GPU2 summary: ", summary_gpu2)
```

On different GPU models, even with the same model and seed, subtle differences in the floating point calculations may result in *very* slight differences in the outputs. This isn't always observable, and may be difficult to measure precisely, but it underlines the complexities involved in exact reproducibility.

**Example 3: Demonstrating influence of beam search**

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)


def summarize_with_beams(model_name, text, num_beams_list, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if torch.cuda.is_available():
      model = model.cuda()

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}


    summary_texts = []
    for num_beams in num_beams_list:
        summary_ids = model.generate(inputs['input_ids'], max_length=120, num_beams=num_beams, early_stopping=True)
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_texts.append((num_beams, summary_text))
    return summary_texts

text = "This is a piece of text that I want to summarize. The summaries may be slightly different depending on how the beam search operates and parameters are tuned. There are multiple routes possible in the generation process which makes this a good example."

beam_results = summarize_with_beams("t5-small", text, [1, 4, 10], seed=42)
for beams, summary in beam_results:
    print(f"Summary with {beams} beams: {summary}")

```

Beam search, while a useful tool, is not fully deterministic, as it maintains *n* candidate sequences that it iteratively refines based on probabilities.  The exact branching and pruning process may vary slightly run to run, especially as beam size changes, leading to different generated text, even if the underlying model parameters are consistent.

For deeper understanding of these complexities, I'd recommend looking at the research literature on *non-convex optimization* as used in deep learning (e.g., "Deep Learning" by Ian Goodfellow et al., Chapter 8), specifically the sections on stochastic gradient descent. The *Transformer Architecture* paper (Vaswani et al., 2017) is also key for understanding the inherent mechanics of these models. Research papers on *dropout* and its impact on model training and inference (e.g., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" by Srivastava et al.) will further highlight where randomness is introduced. And, finally, to more fully ensure deterministic training, look into specific library options to *disable* all possible sources of non-determinism, if they are available. This is not trivial, though, and often comes with speed penalties, and may not always eliminate *all* non-determinism.

In summary, while a fixed seed provides *some* level of reproducibility, the interplay of stochastic processes during training, inference, hardware variances, and even the intricacies of the model itself mean that subtle variations are nearly always present. Truly deterministic results, while theoretically possible, requires a very high level of control and awareness of every potential source of non-determinism.
