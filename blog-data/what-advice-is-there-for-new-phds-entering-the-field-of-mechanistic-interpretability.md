---
title: "What advice is there for new PhDs entering the field of mechanistic interpretability?"
date: "2024-12-11"
id: "what-advice-is-there-for-new-phds-entering-the-field-of-mechanistic-interpretability"
---

Hey so you're a fresh PhD jumping into mechanistic interpretability huh that's awesome super cool field  lots of open questions  kinda wild west really  First things first congrats on finishing  PhD is a beast  seriously  massive respect

So advice  well  it's a pretty new field  so there's no like one size fits all playbook  but here's what I've picked up  mostly by making tons of mistakes  lol

**1 Embrace the Messiness**  Mechanistic interpretability isn't like  oh we'll just run this algorithm and get a perfect explanation  nope  it's super messy  the models we're looking at are huge complicated things  think brains but made of math  and  we're trying to understand how they *think*  so expect lots of uncertainty  lots of dead ends  lots of  "hmm that's weird" moments  it's the nature of the beast

**2  Develop a Broad Skillset** You need a strong foundation in both machine learning and theoretical computer science  seriously  you need to be able to build models and understand them at a deep level  like  understand the actual math behind the algorithms not just how to use a library  also  depending on the area you focus on  you might need some neuroscience cognitive science  even philosophy  knowledge  its very interdisciplinary

**3 Find Your Niche** Mechanistic interpretability is huge  like  enormously huge  you can't tackle everything at once  find a specific area  maybe you're into understanding attention mechanisms in transformers  or explaining the emergent properties of recurrent networks or maybe probing the internal representations of diffusion models  pick something you find genuinely interesting  and focus your energy there  otherwise you'll spread yourself too thin

**4 Collaborate  Collaborate Collaborate** This field thrives on collaboration  seriously  you'll need people with different skills  different perspectives  even different programming languages  to make real progress  find people you can bounce ideas off of  and who can help you when you're stuck  which will happen often  I promise

**5 Build Your Intuition**  This is hard to teach  but it's crucial  you need to develop a gut feeling for what's going on inside these models  what might be important  what might be noise  it comes from experience  from reading papers  from building models  from breaking them  from talking to people  from lots and lots of trial and error

**6 Get Comfortable with the Unknown**  Like I said  this is a new field  there's no established methodology  no guaranteed path to success  you'll have to be comfortable with uncertainty  with ambiguity  with not knowing  a lot  embracing the open questions  that's where the fun is  right


**7  Learn to Communicate Effectively**  You'll need to communicate your findings to both technical and non technical audiences  practice explaining complex ideas simply  practice writing clear concise papers  practice giving engaging presentations  this is super crucial for getting your work out there and making an impact


**8  Don't Be Afraid to Fail**  Failure is part of the process  it's how you learn  don't let setbacks discourage you  learn from your mistakes and move on  everyone fails  even the most successful researchers  honestly its probably how they learned the most

**9 Read Widely** Don't just focus on the latest papers  read classic papers too  understand the historical context of the field  get a sense of how things have evolved  see what approaches have worked and what hasn't  this stuff is crucial for building your intuition

**10  Get Involved in the Community** Attend conferences  go to workshops  present your work  network with other researchers  build relationships  the mechanistic interpretability community is relatively small  but it's very active and supportive  getting involved is a great way to learn  get feedback  and find collaborators



Okay now for some code examples cause you asked for it

**Example 1: Probing Neuron Activation**

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

# Example input image (replace with your own)
image = torch.randn(1, 3, 224, 224)

# Get activations of a specific layer (e.g., layer 4)
activation = model.layer4(image)

# Analyze activations (e.g., calculate mean activation)
mean_activation = torch.mean(activation)

print(f"Mean activation of layer 4: {mean_activation}")
```

This is super basic  just shows how to grab activations from a specific layer  you'd want way more sophisticated analysis  like  looking at individual neuron activations  or using techniques from  "Deep Learning" by Goodfellow Bengio and Courville


**Example 2:  Activation Maximization**

```python
import torch
import torchvision.models as models
import torch.nn.functional as F

model = models.resnet18(pretrained=True)
model.eval()

# Target neuron to maximize
target_neuron = 10

# Optimization loop
optimizer = torch.optim.Adam([image], lr=0.1)
for i in range(100):
    optimizer.zero_grad()
    output = model.layer4(image)
    loss = -output[0, target_neuron] # Maximize activation
    loss.backward()
    optimizer.step()

# Display or analyze the resulting image

```

This is another basic example  you'd use this to see what kind of input image makes a specific neuron fire strongly  again a starting point  read "Interpretable Machine Learning" by Christoph Molnar


**Example 3:  Simple Attention Visualization**


```python
import torch
import matplotlib.pyplot as plt
import transformers

model_name = "bert-base-uncased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModel.from_pretrained(model_name)

text = "This is a test sentence"
inputs = tokenizer(text, return_tensors="pt")

outputs = model(**inputs)
attention = outputs.attentions[0][0] # Get first attention head

#Visualize attention weights (very simplified)

plt.imshow(attention.detach().numpy())
plt.show()
```

This shows how to extract attention weights from a transformer model  super basic visualization  you'd want to do much more  like  plotting attentions across words  or maybe using  more advanced visualization techniques check out papers on attention mechanisms in transformers

Remember these are just tiny snippets  real mechanistic interpretability work is much more complex  and involves a lot more than just code you need theory  intuition  and  a lot of patience  good luck  you'll need it  but it's going to be an awesome journey
