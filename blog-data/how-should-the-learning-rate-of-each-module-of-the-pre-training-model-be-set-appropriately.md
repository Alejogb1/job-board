---
title: "How should the learning rate of each module of the pre-training model be set appropriately?"
date: "2024-12-14"
id: "how-should-the-learning-rate-of-each-module-of-the-pre-training-model-be-set-appropriately"
---

alright, so you're asking about how to fine-tune a pre-trained model and, specifically, how to handle learning rates for different parts of it. i've been down this rabbit hole a few times, and it's a classic problem, especially when you're dealing with something like, say, a large language model or a complex convolutional network. let's break it down.

first off, think of a pre-trained model as something you've partially assembled, like a partially constructed lego castle. someone else built the base, the foundations, and maybe some of the basic towers. that's your pre-training on a huge dataset. now you have your specific task, like building a specific style of roof or adding a unique kind of decorative wall. you want to fine-tune the existing structure to match your specific needs. that's your fine-tuning on a smaller, task-specific dataset.

the core issue here is that not all parts of the pre-trained model are created equal in relation to your task. early layers in, say, a convolutional network are often learning general features like edges, corners, and basic textures, things that are broadly useful. later layers are often more attuned to the specifics of the task the model was pre-trained on (e.g., classifying image net categories). when you fine-tune, you probably want to avoid messing too much with those early layers. they have learned a lot of useful, robust stuff that you probably want to keep around. over-aggressively changing them can degrade their quality and hurt your specific goal. but the later layers might need a more drastic update since they were trained for a different task than you have now.

that's where differential learning rates come into play. the idea is to use smaller learning rates for the initial layers (the base of your lego castle) and larger learning rates for the layers closer to your output (the roof and fancy decorative wall) during fine-tuning.

now, how to implement that? it depends a bit on the framework you're using. let's say we're using pytorch, because i'm most comfortable with it. let’s talk about my personal experience, a few years ago, i was working on a project related to fine-tuning a transformer model for sentiment analysis. the pre-trained model was the typical bert-base-uncased, and at the beginning, i naively used a single learning rate for the entire model. the performance was not terrible, but it was not great neither, i noticed that it was overfitting quickly to the training set and not generalizing well to unseen data. i also noticed some layers were changing way more than i anticipated or liked. so i started to investigate about differential learning rates. i looked into the 'ulmfit' paper, which has a detailed discussion of using gradual unfreezing and differential learning rates. this paper is an excellent starting point if you are interested in the topic, i really recommend it. i read it more than once. and after a few iterations i got the following code snippet working:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

def get_parameter_groups(model, lr_mults):
  """
  splits the model parameters into groups, each with a different learning rate multiplier.
  Args:
    model (nn.module): the model to split.
    lr_mults (list): a list of learning rate multipliers. each element corresponds to a
            group of layers in the model. should be ordered from the bottom to the top.
  Returns:
    list: a list of dictionaries, where each dictionary contains the parameters and lr
            to be used in the optimizer.
  """
  groups = []
  base_params = [] # parameters of the initial layers, the base lego castle
  specific_params = []  # parameters of the later layers, the specific roof
  
  if hasattr(model,'base_model'):
      base = model.base_model
  elif hasattr(model,'bert'):
      base = model.bert
  elif hasattr(model,'transformer'):
      base = model.transformer
  else:
     raise Exception("no base or transformer part in the model") 

  for name, param in base.named_parameters():
    base_params.append(param)

  if hasattr(model,'classifier'):
    for name, param in model.classifier.named_parameters():
        specific_params.append(param)
  elif hasattr(model,'pooler'):
      for name, param in model.pooler.named_parameters():
        specific_params.append(param)
  else:
    raise Exception("no classifier or pooler in the model")


  # all parameters apart of base and head
  extra_params = []
  for name, param in model.named_parameters():
    if not any([param is p for p in base_params]) and not any([param is p for p in specific_params]):
        extra_params.append(param)


  groups.append({'params': base_params, 'lr': lr_mults[0]}) # base layers lr_mults[0] times base_lr
  groups.append({'params': extra_params, 'lr': lr_mults[1]}) # extra layers lr_mults[1] times base_lr
  groups.append({'params': specific_params, 'lr': lr_mults[2]}) # head layers lr_mults[2] times base_lr

  return groups


if __name__ == '__main__':
  model = AutoModel.from_pretrained('bert-base-uncased')

  # define the head, something that is specific to our task
  class classifier(nn.Module):
    def __init__(self, input_size, num_classes):
      super(classifier,self).__init__()
      self.fc1 = nn.Linear(input_size, 128)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.fc2(x)
      return x
  
  model.classifier = classifier(768, 2) # 768 is the output of bert-base-uncased and 2 the number of classes

  # define our custom learning rate multipliers. the lr for base_model (bert), extra_params (no exist in our case), classifier
  lr_mults = [0.1, 0.5 ,1.0]
  base_lr = 1e-5  # base learning rate, a good starting point for bert fine-tuning
  params_to_optimize = get_parameter_groups(model, [base_lr*mul for mul in lr_mults]) # apply multipliers to base_lr
  optimizer = optim.AdamW(params_to_optimize) # use AdamW for bert fine tuning

  # training loop below with the fine-tuned learning rates (just an example, no data)
  for epoch in range(1):
    # simplified training step
    fake_input = torch.randn(2, 10, 768)
    fake_target = torch.randint(0, 2, (2,))
    outputs = model(inputs_embeds = fake_input).last_hidden_state[:,0,:]
    outputs = model.classifier(outputs)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs,fake_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print("done training with differential learning rates")

```

in this snippet, `get_parameter_groups` divides the model’s parameters into groups, and each group has its own learning rate. the base bert layers have a learning rate of `base_lr*0.1` (1e-6), the extra layers have lr = `base_lr*0.5` (5e-6) and the classifier layers with lr = `base_lr * 1` (1e-5). the idea is to keep the base layers (the foundation of the lego castle) more stable while adjusting the classification layers with a bigger learning rate (the roof). also, remember that you can play with the learning rates multipliers of course, 0.1,0.5,1.0 are just an example, play with them. i found that small multipliers like 0.1 for bert layers and around 1.0 for my new layers works quite well for text classification tasks.

i also use adamw optimizer because is frequently recommended for bert and similar models, although you can experiment with other options of course. notice, that i have removed the embedding layers of the bert model from the base parameters, and also the pooler (used for classification), they are kept in specific_params because the initial layers (like the embedding) can require even smaller learning rates than the rest of the base.

 another important technique to consider is gradual unfreezing, which is usually paired with differential learning rates. initially, you freeze most of the pre-trained model layers, training only the newly added ones (the custom task-specific layers), usually with a bigger lr. then you gradually unfreeze some of the pre-trained layers and apply differential learning rates. in the past, i've seen that this works like a charm. you can think about it as building your lego castle step by step, you do not start adding bricks at the top without any previous support. you build the base first and slowly add more elements to the top. i've read about this in the fastai library documentation and in several papers of fastai research group. i recommend checking them.

here's the code implementing the 'gradual unfreezing' technique with the previous example.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

def get_parameter_groups(model, lr_mults, freeze_layers):
    """
    splits the model parameters into groups, each with a different learning rate multiplier.
    Args:
      model (nn.module): the model to split.
      lr_mults (list): a list of learning rate multipliers. each element corresponds to a
              group of layers in the model. should be ordered from the bottom to the top.
      freeze_layers(bool) if true freeze the initial layers, otherwise do not freeze.
    Returns:
      list: a list of dictionaries, where each dictionary contains the parameters and lr
              to be used in the optimizer.
    """
    groups = []
    base_params = [] # parameters of the initial layers, the base lego castle
    specific_params = []  # parameters of the later layers, the specific roof
    
    if hasattr(model,'base_model'):
        base = model.base_model
    elif hasattr(model,'bert'):
        base = model.bert
    elif hasattr(model,'transformer'):
        base = model.transformer
    else:
        raise Exception("no base or transformer part in the model") 

    for name, param in base.named_parameters():
        if freeze_layers:
            param.requires_grad = False #freeze base layers
        base_params.append(param)

    if hasattr(model,'classifier'):
        for name, param in model.classifier.named_parameters():
            specific_params.append(param)
    elif hasattr(model,'pooler'):
        for name, param in model.pooler.named_parameters():
            specific_params.append(param)
    else:
        raise Exception("no classifier or pooler in the model")

    # all parameters apart of base and head
    extra_params = []
    for name, param in model.named_parameters():
        if not any([param is p for p in base_params]) and not any([param is p for p in specific_params]):
            extra_params.append(param)


    groups.append({'params': base_params, 'lr': lr_mults[0]}) # base layers lr_mults[0] times base_lr
    groups.append({'params': extra_params, 'lr': lr_mults[1]}) # extra layers lr_mults[1] times base_lr
    groups.append({'params': specific_params, 'lr': lr_mults[2]}) # head layers lr_mults[2] times base_lr

    return groups

if __name__ == '__main__':
    model = AutoModel.from_pretrained('bert-base-uncased')
    
    # define the head, something that is specific to our task
    class classifier(nn.Module):
      def __init__(self, input_size, num_classes):
        super(classifier,self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
      def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    model.classifier = classifier(768, 2) # 768 is the output of bert-base-uncased and 2 the number of classes

    # first stage: train the last layers (head) only. 
    lr_mults = [0.0,0.0,1.0] # lr = 0 for base, 1 for classifier
    base_lr = 1e-5
    params_to_optimize = get_parameter_groups(model, [base_lr*mul for mul in lr_mults],freeze_layers=True)
    optimizer = optim.AdamW(params_to_optimize)

    #training loop stage 1
    for epoch in range(1):
        #simplified training step
        fake_input = torch.randn(2, 10, 768)
        fake_target = torch.randint(0, 2, (2,))
        outputs = model(inputs_embeds = fake_input).last_hidden_state[:,0,:]
        outputs = model.classifier(outputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs,fake_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("done training stage 1, finetuning the last layers")

    # second stage: train all layers using differential learning rates, unfreeze all layers
    lr_mults = [0.1, 0.5 ,1.0]  # lr for base, extra, classifier
    params_to_optimize = get_parameter_groups(model, [base_lr*mul for mul in lr_mults],freeze_layers=False)
    optimizer = optim.AdamW(params_to_optimize)

    # training loop stage 2
    for epoch in range(1):
        #simplified training step
        fake_input = torch.randn(2, 10, 768)
        fake_target = torch.randint(0, 2, (2,))
        outputs = model(inputs_embeds = fake_input).last_hidden_state[:,0,:]
        outputs = model.classifier(outputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs,fake_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("done training stage 2, finetuning all layers with differential learning rates")
```

in this second code snippet, we have two training stages. in the first one, we freeze the base bert model, only training the last layers. after that we unfreeze the bert model and train all the layers with differential learning rates. this technique usually improves performance. i hope you can see how useful this approach is. also, remember that this is just an example, you can have more than two training stages.

finally, a word on how to pick these learning rates. it's part art, part science. the base learning rate needs some experimentation, and the differential learning rates multipliers require a lot more experimentation, but it also depends a lot on your problem and your specific pre-trained model. i've found that starting with smaller learning rates and gradually increasing them (and multipliers) as the model gets more specialized works quite well, this is based on experience and a lot of trial and error. also remember that using learning rate schedulers are essential. also you need to have the intuition of how these parameters change with the training of the model. i have worked with learning rates since early in my career when i was training neural networks by hand in matlab and it required understanding the maths behind the gradient descent. these days i use pytorch and other libraries of course, but those intuitions are still very valuable. oh, by the way, why don't scientists trust atoms? because they make up everything.

to sum up, you'll likely need to do some experimentation. always plot the learning curves and validate your model with an independent dataset. check out papers like 'ulmfit' and fastai documentation, those resources are great.

hope this helps. let me know if you have any more specific questions.
