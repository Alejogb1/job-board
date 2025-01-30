---
title: "How can I obtain a probability distribution over tokens predicted by a Hugging Face model?"
date: "2025-01-30"
id: "how-can-i-obtain-a-probability-distribution-over"
---
The core of obtaining a probability distribution over tokens from a Hugging Face Transformers model lies in understanding the model's output and leveraging the provided tokenization utilities. The models themselves don't directly output probabilities; they produce logits – raw, unnormalized scores representing the model's confidence in each possible token. Transforming these logits into a probability distribution requires applying a softmax function.

Having spent considerable time fine-tuning language models for various text generation tasks, I've found that correctly interpreting this output is critical. The process isn’t as straightforward as simply calling a `predict()` method. Instead, we must interact with the model’s inference pipeline at a lower level. The key components involved are the model, the tokenizer, and the appropriate processing functions. We utilize the tokenizer to convert raw text into a format the model understands (numeric IDs), feed these into the model, and then interpret the logits using softmax to derive probabilities.

Let’s delve into the practical steps. Assuming you have a Hugging Face Transformers model and tokenizer loaded, the first crucial action is tokenization. This step encodes your input text into numerical representations the model can process. These tokens will be what the model predicts probabilities over.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Example using a GPT-2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "The quick brown fox"
inputs = tokenizer(input_text, return_tensors="pt")
```

Here, `AutoTokenizer` automatically loads the appropriate tokenizer for the specified model, ensuring our input text is encoded in a way the model expects. The `return_tensors="pt"` argument ensures the output is a PyTorch tensor, making it compatible with our model. The tokenizer produces several outputs, but we primarily need the `input_ids` which represent the numerical token representations of the input text.

Next, we feed these input tokens into the model. The model doesn't directly provide probabilities, so we must extract the logits and process them. It's vital to understand that language models, particularly causal ones like GPT-2, produce logits for the next token *given* the input tokens.

```python
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
```
We use `torch.no_grad()` to disable gradient calculations for the inference process, increasing efficiency. The model output structure depends on the specific model, but generally, `outputs.logits` contains the raw scores for each potential token in the model's vocabulary. The shape of the logits is usually `[batch_size, sequence_length, vocab_size]`. Since we're working with a single input, our batch size is 1, and we're generally interested in the logits for the *next* token, not for the provided input sequence. Therefore, to get probabilities of the next token, we examine the logits at the last token in the sequence. For instance, for an input of length *n*, we'd look at `logits[0, n-1, :]`.

Now, to convert these logits into a probability distribution, we apply the softmax function. Softmax normalizes the logits such that they sum up to 1, representing a valid probability distribution.

```python
last_token_logits = logits[:, -1, :]
probabilities = F.softmax(last_token_logits, dim=-1)
```

`F.softmax(last_token_logits, dim=-1)` applies the softmax function to the last dimension, which corresponds to the vocabulary size, creating the probability distribution over all possible tokens. At this point, we have the desired probability distribution. We can access the probability of each token by indexing into the `probabilities` tensor. To get the token corresponding to those probabilities, we use `tokenizer.convert_ids_to_tokens()`.

```python
predicted_token_id = torch.argmax(probabilities, dim=-1) # Find the most probable next token ID
predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id) # Convert the predicted ID to a token
print(f"Next most probable token: {predicted_token}")

top_k_probs, top_k_ids = torch.topk(probabilities, 5, dim=-1) # Find the top 5 probable tokens
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids[0]) # Convert the top 5 token IDs to tokens
print(f"Top 5 most probable tokens: {top_k_tokens}")
print(f"Top 5 probabilities {top_k_probs}")
```

In this example, we've calculated the probability distribution and then identified both the single most probable token and the top 5 most probable tokens and their respective probabilities. This allows us to understand the model's confidence in different predictions.

Let's consider a more complex scenario with different tokenizers. The process is similar, but subtle nuances exist. Consider using BERT-based model as it does not generate text; rather, we can use a masked token to get a probability distribution over the vocabulary of that single masked position.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Example using a BERT model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

input_text = "The quick brown [MASK] jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
mask_token_logits = logits[:, mask_token_index, :]
probabilities = F.softmax(mask_token_logits, dim=-1)

predicted_token_id = torch.argmax(probabilities, dim=-1)
predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
print(f"Next most probable token: {predicted_token}")


top_k_probs, top_k_ids = torch.topk(probabilities, 5, dim=-1)
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids[0])
print(f"Top 5 most probable tokens: {top_k_tokens}")
print(f"Top 5 probabilities {top_k_probs}")
```
This code demonstrates a BERT model, which uses masked token for prediction. We identify the location of the `[MASK]` token, extract logits corresponding to it and calculate probabilities for that token only. This allows us to determine the most probable word to fill the mask token, which is different than predicting the next token for causal models.

Finally, consider a scenario using a sequence-to-sequence model, for example using a T5 model for text summarization. The process is similar to GPT-2, however, we must use an input and decoder ids.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Example using a T5 model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

input_text = "summarize: The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
logits = outputs.scores[0]

probabilities = F.softmax(logits, dim=-1)

predicted_token_id = torch.argmax(probabilities, dim=-1)
predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
print(f"Next most probable token: {predicted_token}")


top_k_probs, top_k_ids = torch.topk(probabilities, 5, dim=-1)
top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids)
print(f"Top 5 most probable tokens: {top_k_tokens}")
print(f"Top 5 probabilities {top_k_probs}")
```
Here we've incorporated the `output_scores=True` and `return_dict_in_generate=True` into the model's generate function. The `outputs.scores` list, instead of `logits`, will return the scores for each token at each position in the generation process. This is key to getting a probability for all tokens in the sequence. In this example we're pulling the scores/logits for the first generated token and computing its probability distributions.

In summary, accessing probabilities over tokens involves tokenization, model inference, extracting logits, and applying the softmax function. The approach can vary slightly based on model architectures, especially between causal, masked, and sequence-to-sequence models.

For further exploration, I recommend investigating documentation and tutorials on the Hugging Face Transformers library, particularly sections covering model outputs, tokenization, and the use of the `softmax` function. Texts focusing on deep learning and neural network architectures can also offer deeper insights into the theoretical basis for the model operations. I’ve also found great value in exploring the numerous examples within the Hugging Face official GitHub repository. These resources combined provide a robust foundation for effectively using and understanding these models.
