---
title: "Fine-tuning LLMs: Avoiding Common Pitfalls"
date: "2024-11-16"
id: "fine-tuning-llms-avoiding-common-pitfalls"
---

yo dude so i just watched this killer talk about squashing bugs in these open-source language models like llama 3 and gemini it was pretty epic honestly  the whole point was to help peeps avoid common pitfalls when they're fine-tuning these massive models  think of it as a supercharged debugging session for ai nerds  it went on for like 20 minutes but i got the gist which i'll spill to you now

first off the speaker mentioned he'd already done a three-hour workshop on this stuff so i'm guessing this was more of a recap and a quick hit list of the most common problems   he mentioned some slides being at some tinyurl address but i'll just give you the highlights

one of the things that really stood out was the whole 'double bos token' debacle the dude was *serious* about this  he kept repeating that you absolutely *must not* use two beginning-of-sentence tokens when fine-tuning llama 3 it turns out that's a super common mistake and it basically wrecks your accuracy during inference he showed this chart of embeddings that went completely haywire when the double bos tokens were present  it looked like total chaos which was pretty funny

i mean this is like putting two batteries in backwards in a flashlight yeah it's obviously gonna screw up your light and the model is basically a super complex flashlight so no double bos  he even showed this simple thing using hugging face's apply chat template:

```python
# good
from transformers import pipeline

generator = pipeline('text-generation', model='your_llama_model')
prompt = "<s>This is a test" # <s> adds the single BOS token
output = generator(prompt, max_length=50)
print(output)

# bad — adds extra BOS tokens — avoid this
from transformers import pipeline

generator = pipeline('text-generation', model='your_llama_model')
prompt = "<s><s>This is a test" # double <s> is a BIG NO-NO
output = generator(prompt, max_length=50)
print(output) # probably garbage or just nothing at all

```

another big problem he highlighted was using the llama 3 base model with the llama 3 instruct template  apparently the base model has a bunch of untrained tokens specifically those reserve tokens with ids 0-250  those tokens are basically placeholders and have zero embeddings resulting in nan values (not a number) during gradient calculations  this totally screws up your training and your model will just fall apart  he showed a graph of the embeddings that was really eye-opening - some tokens at zero and some all over the place.  imagine trying to build a tower of lego bricks, some of them just flat and some super misshapen!

to fix this he suggested either using the instruct version of llama 3 or taking the mean of the *trained* embeddings and using that for the untrained ones  this is tricky though  you gotta mask out those untrained tokens first otherwise your average will be all wrong  he showed some code for that but let me simplify it for you

```python
import numpy as np

# imagine this is your embedding matrix
embeddings = np.random.rand(1000, 768) # 1000 tokens, 768 dimensions
# let's say tokens 0-100 are untrained
trained_embeddings = embeddings[101:]

# calculate mean of trained embeddings
mean_embedding = np.mean(trained_embeddings, axis=0)

# set untrained embeddings to the mean
embeddings[:101] = mean_embedding

print(embeddings)
```

and get this the pad and eos (end-of-sentence) tokens absolutely *cannot* be the same  if they are your model will generate text forever  think of it like a runaway train with no brakes  he said that some models like gpt-3 had this problem too which surprised me but hey its all about checking that your eos and pad token ids are different before you start fine-tuning

then he talked about exporting the fine-tuned model to formats like ggml and llama.cpp  apparently you gotta make sure the chat template you use matches exactly how you fine-tuned it   he said it was a huge pain before because you had to manually do it but his tool unso handles all of that automatically  that's amazing

```python
# (simplified) example of how unso might handle this
# this is pseudocode because i can't reproduce the real unso internals
from unso import export_model

model_path = "path/to/your/fine-tuned/model"
output_format = "ggml"
chat_template = "instruction: {instruction}\noutput: {output}" # example

exported_model = export_model(model_path, output_format, chat_template)
# exported_model is now ready to use with ggml or llama.cpp
```

the speaker then mentioned a few community contributions  one was that you can only use CPU conversion for llama.cpp and that gpu conversion sometimes had problems with precision differences  which makes sense because cpu and gpu do floating-point calculations differently another contribution was that adding system prompts can improve performance  who knew

the whole presentation culminated in showing his collab notebook which is amazing  it uses his open-source library  unso  to simplify the whole fine-tuning process.  basically the unso library handled most of the common problems and had options for  long context fine-tuning by offloading gradients to system memory (not disk) which is crucial for speed he stressed to use 4-bit quantization (load_in_4bit = True) to save tons of memory which is helpful on free collab notebooks  he showed how to adjust hyperparameters like rank and alpha in low-rank adapters and highlighted the importance of fine-tuning all the necessary linear layers (qkv, down, up, and gate)  and he even talked about merging columns in datasets to fit into common chat templates which was an unexpected detail but makes sense

overall the guy was a total pro and super chill which was nice he kept repeating the same important points over and over which was a bit repetitive but i guess that's a good thing for retaining the info he showed off how his library unsoft automates many of these steps that can cause headaches for beginners.  he even had a neat collab notebook for llama fine-tuning using your own data  the dude seriously saved tons of time and effort for people fine-tuning language models that's a huge win honestly
