---
title: "How to Control Large Language Models: Latent Space Manipulation"
date: "2024-11-16"
id: "how-to-control-large-language-models-latent-space-manipulation"
---

dude so this talk was *amazing*  this chick luisa she's like a wizard with these ai models especially the embedding thing  basically she's all about making these super complex models easier to understand and use  think of it as translating alien gibberish into plain english and then letting you *play* with the english version


the whole point of the video is showing how we can get a better grip on what's happening inside these massive language models instead of just throwing prompts at them like darts at a dartboard and hoping something sticks  she calls it steering a car from the backseat with a pool noodle  hilariously accurate tbh


ok so key moments man oh man first she starts with this super relatable analogy about prompting models  it's like trying to control a monster truck with a tiny remote  you kinda have some influence but it’s indirect and messy  this sets up the whole problem she’s tackling getting more direct control


then comes the *latent space*  she explains it like this imagine each word or image gets transformed into a bunch of numbers a vector a point in this crazy high-dimensional space  that space is the latent space  and the location of each point reflects the model's understanding of that word or image


this is where it gets really cool she shows how she can manipulate these vectors directly  imagine you have a sentence about a sci-fi novel "diaspora" she feeds it to her model gets a vector then *tweaks* that vector  and the tweaked vector gets decoded back into a slightly different sentence  sometimes shorter sometimes with a different tone


that's the third key moment  she shows this in action with some amazing demos  one demo is blurring the vector which makes the generated text fuzzier and less precise  like a semantic version of a photo filter


then fourth she demonstrates how to move the vector along specific directions  she has calculated directions in this latent space that correlate with things like sentence length or sentiment  so she pushes the vector along the "shorter sentence" direction and boom she gets shorter sentences about the same topic


finally fifth she shows how she can mix embeddings  she takes two sentences and combines their vectors creating a new vector  decoding this hybrid vector produces text that sort of blends the two original sentences it's like a textual dna cocktail 


let’s dive deeper into a couple concepts  first *embeddings*  these are basically numerical representations of something like words images or even entire sentences each embedding is a vector of numbers usually very high dimensional  think 1024 or even 2048 numbers  these numbers capture the essence of what’s being represented  the relationships between different embeddings reveal relationships between the things they represent


here’s a simple python example using sentence-transformers a library that makes embedding super easy


```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

sentences = [
    "this is an example sentence",
    "each sentence is converted into a vector",
    "these vectors are called embeddings"
]

embeddings = model.encode(sentences)

print(embeddings) # you’ll see a bunch of numbers this is your embedding

# now you can calculate cosine similarity to see how close sentences are
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)
```


this code first loads a pre-trained embedding model  then it converts three sentences into their respective embeddings  finally it calculates the cosine similarity between the embeddings  a higher similarity score indicates that the sentences are more semantically similar


another crucial concept is *latent space manipulation*  this is where luisa’s magic happens  she doesn’t just create embeddings she actively changes them  she moves vectors along specific directions to change the characteristics of the output  this allows for fine-grained control over the model’s output  something that standard prompting just can’t achieve


here's a conceptual code snippet illustrating this  it's a simplified representation of the process because the actual manipulations are much more complex


```python
import numpy as np

# assume we have an embedding vector 'original_embedding'

# let's say we found a direction vector 'length_direction' that reduces text length

# control parameter
scaling_factor = 0.5

# manipulate the embedding by adding length_direction scaled by our factor
modified_embedding = original_embedding + scaling_factor * length_direction

# decode the modified embedding  this part is super complex usually done by a large model
decoded_text = decode_embedding(modified_embedding)
print(decoded_text) #expect shorter text
```


this code isn't a complete solution it's a simplified visualization  the `decode_embedding` function is a black box representing the complex decoding process used by the model  the key is the addition of the `length_direction` vector which alters the `original_embedding` changing the output  it's essentially a vector addition in the latent space


and get this she even shows how she adapted her model to decode embeddings from *other* models like openai's models  it's like she built a universal decoder a translator for all these different ai languages super cool  this is shown in the video by embedding text in an openai model and then successfully reconstructing it using her model


```python
# in reality this requires lots of training data and a specialized adapter
# but conceptually:
import numpy as np

openai_embedding = get_openai_embedding("this is a test")

# linear adapter trained to map openai embeddings to luisa's model's embeddings
adapted_embedding = linear_adapter(openai_embedding)

# then decode using luisa's model
decoded_text = decode_embedding(adapted_embedding)

print(decoded_text) # we hope the decoded text is close to the input
```

this showcases the power of adapting her model  the `linear_adapter` is the key it learns the mapping between different embedding spaces  this adaptability is a huge step towards interoperability between different ai systems


basically the resolution is that by directly manipulating the latent space of these models we can gain a lot more control and understanding  this gives us more intuitive ways to interact with these powerful models and unlocks a whole new level of possibilities  think of it as getting behind the wheel instead of just honking the horn from the backseat


so yeah  it’s a mind-blowing talk  she makes this insanely complex stuff accessible and fun  i'm totally inspired to dive deeper into embeddings and latent space  maybe i’ll even try to build some of her tools it's definitely worth checking out the hugginface models she mentioned  happy coding my friend
