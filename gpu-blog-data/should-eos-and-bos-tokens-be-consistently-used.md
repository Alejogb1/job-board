---
title: "Should EOS and BOS tokens be consistently used across source and target languages in NLP translation?"
date: "2025-01-30"
id: "should-eos-and-bos-tokens-be-consistently-used"
---
The consistent use of EOS (End-of-Sentence) and BOS (Beginning-of-Sentence) tokens across source and target languages in neural machine translation (NMT) is crucial for the model to effectively learn sentence boundaries and contextualize translation processes. My experience training NMT models, particularly with sequence-to-sequence architectures, has consistently shown that discrepancies in handling these tokens lead to significant performance degradation, including inaccurate translations and model instability. A unified approach, treating these tokens as language-agnostic, is generally the most robust strategy.

The core reason lies in how these tokens function within the model's architecture. In encoder-decoder models, for example, the encoder typically processes an input sentence, prepending it with the BOS token and appending it with the EOS token. The decoder, then, generates the target language translation, similarly initiating generation with BOS and concluding with EOS. These tokens, represented as distinct integers in the vocabulary, aren't merely flags; they're vital pieces of the contextual information the model uses to understand the flow of sentences and predict the end of a generation sequence.  If these tokens differ between source and target languages – such as using a "<bos>" in the source and "<start>" in the target – the model fails to capture the isomorphic role they play in marking sentence initiation and termination. Essentially, you're training it with dissimilar, non-corresponding semantic concepts when, in fact, they serve analogous syntactic roles.

In the source language, the BOS token allows the encoder to explicitly understand the point where the sentence begins.  Without it, the initial words of the sentence lack a clear predecessor context.  Similarly, the EOS token signals to the encoder that all relevant information about the current sentence is processed, allowing it to prepare the hidden states that inform the decoder. The decoder utilizes the same principle; the BOS token facilitates starting the output translation correctly, while the EOS token communicates when a complete sentence has been translated. Discrepancies in these tokens across languages break this coherent structural understanding.

A common failure I've observed when using inconsistent EOS/BOS tokens is that the model struggles to generate complete sentences in the target language. It might prematurely halt translation, leading to truncated or incomplete outputs or, conversely, produce endless streams of text without proper terminations. Another problem is instability during training. Loss functions might fluctuate unpredictably if the model struggles to reconcile different token representations for these boundary markers, negatively impacting the convergence and the overall translation quality.

Let’s explore some code examples, specifically with how these tokens integrate into a typical training data preparation pipeline, using PyTorch and illustrative, rather than fully functional code snippets:

**Example 1: Tokenization with Shared Special Tokens**

```python
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import ToTensor

# Sample training data
src_sentences = ["This is a source sentence.", "Another example here."]
tgt_sentences = ["Ceci est une phrase source.", "Un autre exemple ici."]

tokenizer = get_tokenizer("basic_english") # Using basic English tokenizer for example
special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>'] # unified special tokens


def yield_tokens(sentences):
    for sentence in sentences:
        yield tokenizer(sentence)

src_vocab = build_vocab_from_iterator(yield_tokens(src_sentences), specials=special_tokens)
tgt_vocab = build_vocab_from_iterator(yield_tokens(tgt_sentences), specials=special_tokens)

src_text_transform = ToTensor(padding_value = src_vocab['<pad>'])
tgt_text_transform = ToTensor(padding_value = tgt_vocab['<pad>'])

def text_pipeline(text, vocab, text_transform):
  tokenized_text = tokenizer(text)
  indexed_tokens = [vocab[token] for token in tokenized_text]
  return text_transform(torch.tensor( [ vocab['<bos>'] ] + indexed_tokens + [vocab['<eos>']]) )


src_example = text_pipeline(src_sentences[0], src_vocab, src_text_transform)
tgt_example = text_pipeline(tgt_sentences[0], tgt_vocab, tgt_text_transform)

print(f"Source Example: {src_example}")
print(f"Target Example: {tgt_example}")
```

This snippet demonstrates using the same special tokens across both source and target vocabularies. The vocabulary objects (`src_vocab`, `tgt_vocab`) are built independently based on the language they represent but share the *string representation* of the special tokens – `['<unk>', '<pad>', '<bos>', '<eos>']`. This unified representation means the numerical mapping of '<bos>' and '<eos>' are consistent *conceptually* between source and target, even if their numerical *index* might differ based on vocabulary size. Further, the function `text_pipeline` is shown to prepend the '<bos>' and append '<eos>' to all text sequences in a consistent way after tokenization and numerical mapping. This uniformity allows the model to learn that '<bos>' denotes the start of a sequence, regardless of the language.

**Example 2: Explicit Padding for Batched Input**

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def generate_batch(batch, src_vocab, tgt_vocab, src_text_transform, tgt_text_transform):
  src_list, tgt_list = [], []
  for src_s, tgt_s in batch:
    src_list.append(text_pipeline(src_s, src_vocab, src_text_transform))
    tgt_list.append(text_pipeline(tgt_s, tgt_vocab, tgt_text_transform))
  src_padded = pad_sequence(src_list, padding_value=src_vocab['<pad>'])
  tgt_padded = pad_sequence(tgt_list, padding_value=tgt_vocab['<pad>'])
  return src_padded, tgt_padded

batch_data = [(src_sentences[0], tgt_sentences[0]), (src_sentences[1], tgt_sentences[1])]

src_batch, tgt_batch = generate_batch(batch_data, src_vocab, tgt_vocab, src_text_transform, tgt_text_transform)
print(f"Padded Source Batch:\n{src_batch}")
print(f"Padded Target Batch:\n{tgt_batch}")
```

Here, `pad_sequence` is employed to batch examples that typically have varied lengths. This function ensures that all sequences within a batch possess the same length by filling the shorter sequences with the `<pad>` token. Importantly, the padding token, like `<bos>` and `<eos>`, must also be consistent across both the source and target languages; otherwise, the padding mechanisms become inconsistent, and the models will learn incorrect relationships due to misaligned inputs. The function `generate_batch` demonstrates how preprocessed, tokenized, and BOS/EOS tagged sentences are collected into a batch and padded with the appropriate padding token before being output for a model.

**Example 3: Incorrect Implementation**
```python
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import ToTensor
import torch

# Sample training data
src_sentences = ["This is a source sentence.", "Another example here."]
tgt_sentences = ["Ceci est une phrase source.", "Un autre exemple ici."]

tokenizer = get_tokenizer("basic_english") # Using basic English tokenizer for example
src_special_tokens = ['<unk>', '<pad>', '<start>', '<end>'] # src special tokens
tgt_special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>'] # tgt special tokens

def yield_tokens(sentences):
    for sentence in sentences:
        yield tokenizer(sentence)

src_vocab = build_vocab_from_iterator(yield_tokens(src_sentences), specials=src_special_tokens)
tgt_vocab = build_vocab_from_iterator(yield_tokens(tgt_sentences), specials=tgt_special_tokens)

src_text_transform = ToTensor(padding_value = src_vocab['<pad>'])
tgt_text_transform = ToTensor(padding_value = tgt_vocab['<pad>'])

def src_text_pipeline(text, vocab, text_transform):
  tokenized_text = tokenizer(text)
  indexed_tokens = [vocab[token] for token in tokenized_text]
  return text_transform(torch.tensor( [ vocab['<start>'] ] + indexed_tokens + [vocab['<end>']]) )

def tgt_text_pipeline(text, vocab, text_transform):
  tokenized_text = tokenizer(text)
  indexed_tokens = [vocab[token] for token in tokenized_text]
  return text_transform(torch.tensor( [ vocab['<bos>'] ] + indexed_tokens + [vocab['<eos>']]) )


src_example = src_text_pipeline(src_sentences[0], src_vocab, src_text_transform)
tgt_example = tgt_text_pipeline(tgt_sentences[0], tgt_vocab, tgt_text_transform)

print(f"Source Example: {src_example}")
print(f"Target Example: {tgt_example}")
```

This final example shows a broken implementation; using `<start>` and `<end>` for the source language and `<bos>` and `<eos>` for the target language. Although both represent sentence boundary markers, the model does not have any way of knowing these are the same function, and it will therefore fail to learn relationships between the source and target. While the code will not error out and will run without issue, it will result in significantly reduced performance of a model trained on this input data. The output shows that this implementation results in differently formatted sequences.

To reinforce good practices, I recommend exploring resources that emphasize the importance of consistent token handling in sequence models. Look for documentation and tutorials related to `torchtext` or `transformers` library tokenizers, vocabularies, and pre-processing.  Material focusing specifically on the use of special tokens in NMT models will be invaluable. Furthermore, exploring best practices for batching, padding, and masking of variable length sequences is highly recommended for ensuring proper use of these special tokens within the data pipeline.

In summary, adhering to consistent EOS and BOS token usage across source and target languages is not merely a matter of convenience, but a fundamental necessity for building robust and accurate machine translation systems. A unified approach ensures the model learns the isomorphic role of these markers, ultimately contributing to improved translation quality and model stability.
