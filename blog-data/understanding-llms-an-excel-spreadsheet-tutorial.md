---
title: "Understanding LLMs: An Excel Spreadsheet Tutorial"
date: "2024-11-16"
id: "understanding-llms-an-excel-spreadsheet-tutorial"
---

dude so i just watched this insane video this guy isan  – seriously brilliant dude –  breaks down gpt-2 using… wait for it… an excel spreadsheet  🤯  like seriously  he built the whole thing in excel  no python no fancy apis just pure excel formulas  the whole point of the vid was to show how llms actually work under the hood  no more black box magic  he wants everyone to understand  even without a machine learning degree  it's a crash course in llm anatomy  thinking and surgery – all in excel  wild right?


 so the setup was this  we're all ai brain surgeons  gpt-2 small is our patient  and the operating table is a 124 million cell excel spreadsheet – i'm not even kidding  he's got like 150 tabs  each tab representing a different part of the model  the goal is to understand how the thing actually generates text  not just treat it like some magical oracle


one of the first things he shows is tokenization  this is where you break down a sentence into smaller units called tokens   remember that visual of him splitting "funology" into "fun" and "ology"? hilarious  the algorithm isn't perfect though  sometimes it splits words in weird ways like "reinjury" becoming "rain" and "jury" it’s like the algorithm is a little drunk but it gets the job done most of the time  it just grabs the most common subword units it's seen during training  it’s a necessary evil as he put it –  a bit like a slightly clumsy but ultimately effective intern


he then explains embeddings  each token gets mapped to a vector of numbers a really long one – 768 numbers in gpt-2 small’s case  think of it like translating words into a language computers understand – a vector space where similar words are closer together.  you see this visually as he scrolls across hundreds of columns in his excel sheet to show the embedding of a single word like "Mike" – it’s a massive vector


then comes the meaty stuff – the attention mechanism and the multi-layer perceptron (mlp) which is just a fancy name for a neural network.  the attention bit is super cool  tokens “look” at other tokens to figure out context.  remember the example where "he" looked mainly at "Mike" because it's the pronoun’s antecedent? brilliant! that’s context understanding at its core  and you can actually see this in his spreadsheet –  a matrix showing how much attention each token pays to others  it’s like a visual representation of the model's internal thought process. zero means no attention – and no token can see into the future, only the past  the mlp part is just loads of matrix multiplications and additions –  the pure number crunching bit where the magic happens


here's a tiny bit of pseudocode to give you a feel for the attention mechanism – it's simplified but captures the essence


```python
# simplified attention mechanism
def attention(query, key, value):
  # query, key, value are matrices representing token embeddings
  scores = query @ key.T  # matrix multiplication to calculate attention scores
  attention_weights = softmax(scores) # normalize scores into probabilities
  context_vector = attention_weights @ value  # weighted average of values
  return context_vector

#example usage (completely simplified)
query = [[1,2,3],[4,5,6]]
key = [[7,8,9],[10,11,12]]
value = [[13,14,15],[16,17,18]]

context_vector = attention(query,key,value)
print(context_vector)
```

this code snippet is just a tiny part – the actual implementation in gpt-2 is far more complex  but it gives you a glimpse into how attention works – calculating attention scores between tokens and weighting values based on those scores.  it shows the core idea behind context awareness.


he also talks about the layers – 12 of them in gpt-2 small. each one is a tab in the spreadsheet  the information flows through these layers,  like an "information superhighway"  with residual connections acting as bypasses allowing info to skip layers if needed  it’s beautifully explained as a network of communication – brilliant analogy


and get this – he uses a technique called "logit lens"  it's like sticking an MRI machine between each layer to see what the model is thinking at each stage  this visual representation of how the model’s internal understanding of a sentence evolves through the network is just amazing


here's a small code snippet to illustrate the concept of a simple logit lens. Note this is significantly simplified and does not reflect the complexity of a real-world implementation.


```python
import numpy as np

def simple_logit_lens(model, input_text, layer_index):
    #this is a super simplified example. a real logit lens needs access to internal model activations
  # get model predictions at a specific layer (again, simplified)
  layer_activations = model.predict(input_text, layer_index) #simplified prediction function. a real logit lens needs direct access to the network's internal activations 
  #assuming layer_activations are logits
  probabilities = softmax(layer_activations)
  return probabilities

#example use (again, super simplified)
layer_probabilities = simple_logit_lens(model, "today is tuesday tomorrow is", 3)
print(layer_probabilities)
```

this is obviously a massively oversimplified representation of a logit lens.  accessing internal model activations in real life is complex – that's what makes this so impressive in excel.


the final part is AI brain surgery  he's using something called sparse autoencoders –  a technique for figuring out what features the model is focusing on  it then manipulates these features to control the model's output.  remember the jedi example?  by tweaking a vector associated with "jedi" in the residual stream –  he gets gpt-2 to predict "lightsaber" instead of "phone"  this is direct manipulation of the model’s latent space! it's mind-blowing.  and the whole thing runs in excel  


here's a snippet showing the core idea  this is highly simplified of course  the real implementation uses the sparse autoencoder


```python
# simplified feature manipulation
jedi_vector = np.array([0.1, 0.2, 0.3, ...]) # a vector from the sparse autoencoder
coefficient = 2  # amplify the Jedi feature
modified_residual_stream = original_residual_stream + coefficient * jedi_vector
```

this code simply adds a scaled jedi vector to the original residual stream.  a real implementation involves a far more complex interaction with the model's internals but illustrates the core idea of feature manipulation.


the resolution  well it's that you can understand llms without being a rocket scientist  isan’s whole point is to demystify these models –  show they're not magic  they're just complicated math  and  excel spreadsheets.  he also makes the point that understanding how these models work inside is not only helpful for building better AI – but also communicating effectively with non-technical stakeholders –  those stakeholders often see these models as inherently magical and are frequently unaware of the limitations or biases.


so yeah  the video is seriously impressive  it's a deep dive into the inner workings of llms  made incredibly accessible and fun by his excel spreadsheet approach  and his amazing analogies  and his whole presentation is ridiculously engaging  check it out if you ever want a wild ride through the world of language models. it's not just a summary it’s a journey!
