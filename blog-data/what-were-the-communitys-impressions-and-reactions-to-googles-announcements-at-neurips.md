---
title: "What were the community’s impressions and reactions to Google’s announcements at NeurIPS?"
date: "2024-12-12"
id: "what-were-the-communitys-impressions-and-reactions-to-googles-announcements-at-neurips"
---

okay so NeurIPS yeah google's stuff there always gets tongues wagging feels like a yearly ritual doesn't it

last year's show was interesting definitely felt a shift maybe a maturity even i don't know not just flashy demos but real stuff being put out there that people could actually poke at and use more than a few eye rolling demos

the general vibe from the community was pretty diverse you know some really hyped some skeptical some downright confused like always the internet

first off let's talk about the big language models those things dominated conversation everywhere not just at NeurIPS you saw a lot of folks getting excited by the sheer scale of the models and their ability to do things we didn't really imagine possible a few years back like the code generation for example it’s mind-blowing to see how good those models have become but then theres the other side the sustainability argument comes back hard that amount of compute has to hurt someone or at least the planet that made a lot of people uncomfortable

the research papers were the meat and potatoes though tons of deep dives into things like explainable AI and federated learning those were definitely the hot topics of the academic side people seemed to be really digging into the more fundamental aspects of AI finally moving past just throwing more data and processing power at the problem there were fewer “magic” tricks and more focus on what makes it work and also importantly making it less a black box

there was also a really strong interest in fairness and ethics that was pretty clear in the workshops and posters that means people are realizing AI isn't just about accuracy and speed you actually have to think about how these models are affecting people and their lives so that definitely felt like a huge and welcome step forward

then you have the practical application side folks were asking hard questions about how google's tools can be used in the real world not just in labs people wanted to see how their models were robust and can handle messy data and even weirder real-world scenarios the whole debate around bias in datasets kept popping up as always and people were talking a lot about how to mitigate it which is honestly the right direction

the smaller stuff also caught people’s eyes more than before things like efficient model training and compression techniques those are critical for smaller startups or anyone who doesn't have google-level infrastructure people were trying to see how to take those advanced research ideas and scale them down make them accessible to everybody it's all about democratization of tech i think that's really a core theme of these events these days

but its never just sunshine and rainbows right skepticism was definitely there a ton of people questioned the real-world impact of some of the more theoretical research are these breakthroughs actually going to help anyone outside the research community some people were pretty cynical saying that it's more like a “science for scientists” type situation which is a fair point i think

and of course the open source debate is still going strong is google truly open sourcing their models or are they holding back key components that question always seems to float around the room and I think that's something people are always going to question and it's good that they do because being fully open is really hard and also it has real life impacts

then the whole ethical AI question never goes away it's almost always the elephant in the room it came up again and again questions about bias about who benefits and who gets harmed by AI it’s crucial to keep having those conversations but it’s really tough to find real answers because the whole thing is so complex it has many layers and the problem also changes as the tech advances

also a few people were saying some of the work felt like incremental improvements rather than genuine paradigm shifts you know refining previous ideas instead of some radically new concept and that’s valid criticism of course you always want something totally new but building on good stuff is important too and it’s actually where most of the work lives

so to wrap it up it feels like the community's reaction is that google is doing really impressive stuff technically no doubt but it comes with a lot of questions about how we use it how we make it fair how we make it open and accessible and all of that is a good thing because it means people aren't just accepting it at face value they are digging deeper and asking the tough questions and that is always a good thing when it comes to tech

oh before i forget i wanted to throw in some code examples to illustrate a point i was making about open source contributions in machine learning so here is the first one showing how to do some basic model training in a framework that google often contributes to

```python
import tensorflow as tf

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate some dummy data
import numpy as np
data = np.random.random((1000, 10))
labels = np.random.randint(0, 10, (1000,))
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)

# Train the model
model.fit(data, one_hot_labels, epochs=10)
```

this example uses tensorflow its like one of the most common frameworks now for deep learning its really powerful and you can actually see how it becomes really simple to do a lot of complex things in machine learning

then lets say we want to explore how to improve the model performance you know some of the techniques discussed in some papers in the conference like for example model compression methods such as pruning this next piece of code demonstrates a very basic pruning concept using tensorflow model optimization

```python
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np

# assume we have a pre-trained model 'model' from before

# Apply pruning with a target sparsity of 50%
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.0, final_sparsity=0.5, begin_step=0,
          end_step=1000)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)

# Recompile the model
model_for_pruning.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Retrain to fine tune
model_for_pruning.fit(data, one_hot_labels, epochs=5)
model_pruning_strip = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

```

this shows how some of the more complicated or seemingly advanced techniques are actually pretty easily implemented using tools like this and thats important because it brings this kind of tech to many people it is not just for people in very specific labs

finally a quick peek at how federated learning can be coded out using a different framework like `pysyft` which shows how researchers are increasingly getting involved in distributed training that reduces the requirements for huge datasets and also preserves privacy in many situations

```python
import torch
import syft as sy
hook = sy.TorchHook(torch)

# create a couple of virtual workers
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")


data = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
target = torch.tensor([[0.], [1.], [1.], [0.]])

# send data to remote workers
data_bob = data[0:2].send(bob)
data_alice = data[2:].send(alice)
target_bob = target[0:2].send(bob)
target_alice = target[2:].send(alice)


# Define model
model = torch.nn.Linear(2, 1)

# move model to remote worker bob (we will make the model learn on bob’s data)
model = model.send(bob)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def train_model(data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = ((output - target)**2).sum()
    loss.backward()
    optimizer.step()
    return loss

for i in range(10):
    loss = train_model(data_bob, target_bob)
    print(f'loss on bob {loss}')

# after training on bob we can pull back and use it locally or even continue training
model_ptr = model.get()
```

those pieces of code are just to give an idea of how these tools operate but really deep down they are supported by a lot of theoretical knowledge that has been published over many many years

if you are interested to dive deeper in the subject of ethical considerations in AI i recommend looking into resources like "ethics of artificial intelligence" by oxford university press its a collection of essays from leading ethicists and researchers exploring all the challenges we face today

another great source is the book "deep learning" by ian goodfellow et al its a great comprehensive overview of the foundations of deep learning and its really helpful to build a strong base for more complex research

and finally for the open source aspect look into some of the papers published about open source model development its hard to point one specifically but usually you will find great discussion on the `arxiv` website about different models and the challenges of open sourcing them in a way that benefits the community without creating new problems

all that should give you a really good head start into understanding the nuances of the conversations happening at events like neurips and more specifically how google is perceived by this community its all an on going journey and there’s a ton more to explore
