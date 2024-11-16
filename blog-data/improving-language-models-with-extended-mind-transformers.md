---
title: "Improving Language Models with Extended Mind Transformers"
date: "2024-11-16"
id: "improving-language-models-with-extended-mind-transformers"
---

dude so i just watched this awesome talk on extended mind transformers and my brain is still kinda buzzing  it's like this whole new level of making language models way smarter without breaking the bank or making them totally confused

the whole point of the video was showing how to make language models way better at using external knowledge  think of it like giving your chatbot a super-powered memory that it can actually use effectively  most chatbots kinda just shove everything into the prompt which is like, giving a kid a whole library and expecting them to only read the right book – a recipe for disaster

okay so first off there's this super relatable thing she mentions – the problem with current methods  she calls out long context and retrieval augmented generation (RAG)  remember how she says long context is like trying to stuff a whole encyclopedia into the prompt  that's exactly it it's expensive slow and the model gets overwhelmed with irrelevant info  it's like trying to cook a gourmet meal with all the ingredients in a giant pile  

she also points out that RAG, while better, makes decisions about what's relevant *before* the model even starts thinking  it's like picking your ingredients before you even know what you're cooking  this is a visual cue – she shows a graphic of  'pre-generation selection'  it's all very upfront and doesn't let the model make those choices dynamically

then she gets into the meat of it extended mind transformers  the core idea is simple  she mentions "extended mind attention" which is a super clever tweak  instead of shoving everything in the prompt it lets the model *actively retrieve* the info it needs as it's generating text  this is like having a personal research assistant that only grabs the specific books your essay needs

this brings us to our first key idea –  the actual mechanism  it’s all about how the model uses attention  remember, transformers use attention to figure out which parts of the input are important   well, extended mind transformers add a special kind of attention that lets the model pull in information from an external memory as it generates text  it's like giving the model a super-powered search function integrated directly into its brain


here's some pseudo-code to illustrate the attention mechanism part of it

```python
# simplified extended mind attention
def extended_mind_attention(query, keys, values, top_k):
    # calculate similarity between query and keys (cosine similarity for example)
    similarities = calculate_similarity(query, keys)

    # select top k most similar keys
    top_k_indices = tf.math.top_k(similarities, k=top_k).indices

    # retrieve corresponding values
    top_k_values = tf.gather(values, top_k_indices)

    # perform attention using top_k_values
    context = attention_mechanism(query, top_k_values)

    return context 

# placeholder functions
def calculate_similarity(query, keys):
    # your cosine similarity calculation here
    pass

def attention_mechanism(query, values):
    # your attention mechanism logic here
    pass
```

basically the model gets a query, finds the most relevant bits from its memory (the keys and values), and focuses on those for generating its answer.

second key idea is how they handle position information   transformers are usually position-agnostic they don’t inherently know the order of words   but when you pull in extra info from memory, you gotta tell the model where each new piece fits  she mentions two solutions – rotary position embeddings and alibi linear biases –  both ways to give the model a sense of order without much extra training.  she mentions this is much easier now because of advancements in relative position embeddings vs. absolute position embeddings


another crucial part is the citations  she says,  "post-hoc rationalization" is what RAG does you can kind of guess why the model said what it said but you can't really be sure   but with extended mind transformers the model shows you exactly which bits of external knowledge it used  this is a game-changer for trust and explainability

here's a python snippet illustrating how this improved citation could look:

```python
# example showing citation tracking
memories = ["Alexander Grothendieck was born in Berlin", "He became a French citizen in 1971"]
query = "When did Alexander Grothendieck get French citizenship?"
response, retrieved_indices = extended_mind_model(query, memories)

print("Response:", response) # output: "He became a French citizen in 1971"
print("Retrieved Memories:", [memories[i] for i in retrieved_indices]) # output: ["He became a French citizen in 1971"]

```

see how the code shows which memory was used this is crucial for transparency


then there's the active learning bit – hallucination reduction she talks about how if the model is unsure about something it can go back to its memory to get more info this is like the model saying  "hmm i'm not totally sure let me check my notes" it's brilliant and solves the hallucination issue, a common problem in LLMs where they make up stuff

here's an example of that active learning approach in code:

```python
# hallucination reduction using active learning
def generate_with_uncertainty(query, memories, uncertainty_threshold=0.8):
    response, probabilities = model(query, memories) # probabilities is the uncertainty score.

    for i, prob in enumerate(probabilities):
        if prob < uncertainty_threshold:  # if the model is uncertain
            additional_memories = retrieve_additional_memories(response[:i]) # grab more info
            response[i:] = model(query, memories + additional_memories)[0][i:] # regenerate with extra info

    return response
```

it's still pseudo-code but it illustrates the concept of checking for uncertainty and recursively getting more information when needed

finally she talks about tuning parameters  stride length (how much memory you look at each time) and top_k (how many relevant memory items you use) are crucial.  too small a stride is slow too large misses important details too small a top_k will hurt performance  too large risks confusion

the whole thing ends with her showing off their open-source code and models on huggingface and github  and the counterfactual retrieval benchmark they created, which is used to evaluate the model's ability to prioritize the provided information during inference time, rather than relying on memorized facts.  it's a pretty cool way to test these models  she emphasizes that it's surprisingly easy to use  just plug in your memories and go

so the main takeaway this whole thing was  extended mind transformers are a super effective way to boost language model performance  they actively retrieve info they're transparent about what they use they reduce hallucinations and they're relatively simple to implement  this is way beyond just summarizing it really gives you a grasp of the brilliance and potential of this tech  it's pretty mind-blowing stuff and a totally fresh approach to making AI more useful and trustworthy
