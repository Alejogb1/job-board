---
title: "How to Obtain a List of Losses Per Epoch from a FastAI Learner?"
date: "2024-12-14"
id: "how-to-obtain-a-list-of-losses-per-epoch-from-a-fastai-learner"
---

so, you're after getting a hold of the training losses from your fastai learner, huh? i've been there, staring at those metrics during training, wishing i could just pluck those epoch-by-epoch loss values out and do something useful with them. it's a fairly common need when trying to really understand how your model is progressing, spotting potential issues, or just wanting to create detailed visualizations of your training run. let me tell you, i’ve spent a good chunk of time banging my head against the wall figuring this out. i remember back when i was first starting to use deep learning, i was trying to train this image classification model on a very niche dataset i had, and the training curve was all over the place. i thought something was broken, but i couldn’t get a handle on what, because i couldn’t easily see the per-epoch losses and i had to manually take the outputs from the terminal and put them into excel which was pain in the butt.

fastai, thankfully, gives us the tools to do this without too much trouble. there isn't a single property directly holding all the losses neatly packaged, but we can extract them from the `recorder` object within the learner. let's walk through how.

first, the core of it lies in the `recorder` attribute of your learner. this recorder stores the metrics generated during training and validation. inside this recorder, you’ll find the training losses in `recorder.values`.

`recorder.values` contains a list of tuples. each tuple contains the training and validation metrics calculated at the end of an epoch, for instance `[epoch_number, training_loss, validation_loss, metric_one, metric_two]` and so on. since you are just after the training losses, you only want to extract the training loss which happens to be index 1 of that tuple.

here’s a quick code snippet illustrating how you can grab those training losses for each epoch:

```python
def get_training_losses(learner):
    losses = [item[1] for item in learner.recorder.values]
    return losses

# assuming your learner is defined as `learn`
training_losses = get_training_losses(learn)
print(training_losses)
```

that function above, `get_training_losses`, will return a list containing all the training losses that the model saw while being trained.

a word of caution though: if you are using a learning rate finder before training your model and then train your model, there's an important detail to remember. the learning rate finder's calculations actually fill up some portion of the `recorder.values` list *before* your actual training begins. these pre-training values would get included in the output if you don't filter them, so you need to make sure that you only extract values from when training happens. you usually want to avoid that so you should slice the recorder list to only get the losses after the learn rate finder.

one strategy i’ve found pretty robust is to use the `fit` method, which sets up the training loops and also fills the `recorder.values`. so, if you run `learn.lr_find()`, and then `learn.fit(epochs=...)` that will create the expected sequence of recorder entries. after the `.fit()` method completes it records the values and then you can access them. here's how you can do that:

```python
# first find the learning rate
lr_min, lr_steep = learn.lr_find()

# then train the model using the fit method and epochs
learn.fit(10)

def get_training_losses_post_fit(learner):
    losses = [item[1] for item in learner.recorder.values]
    return losses

training_losses = get_training_losses_post_fit(learn)
print(training_losses)

```

this makes sure that the first few values are not from the learning rate finder but directly from the training loop. it prevents your plots from looking odd.

now, let’s say you are training with multiple cycles and you want to get the total loss of all cycles, you will need to concatenate the losses, here is another snippet for that:

```python
def get_training_losses_cycles(learner):
  all_losses = []
  for cycle in range(2):
    learn.fit_one_cycle(5)
    losses = [item[1] for item in learner.recorder.values]
    all_losses = all_losses + losses
  return all_losses


training_losses = get_training_losses_cycles(learn)
print(training_losses)

```

this `get_training_losses_cycles` function trains the model for 2 cycles of 5 epochs each and gets a concatenated list of losses for both cycles in the training process.

getting access to these losses is useful for many things: you can plot how the loss decreases over time. you can detect when you've reached a convergence or if there is some type of instability during training. it lets you diagnose problems more systematically. once i was testing a new optimizer and when i used this technique i realized it had some weird behaviors during the first 10 epochs. without this i would've never noticed it.

also, you might want to combine it with other plots, such as plotting the learning rate on the same x-axis to visually correlate its changes with the loss values. that's another handy trick i’ve learned the hard way. i once spent a whole week trying to figure out why the model wasn't training properly, and turns out that the learning rate was changing way too fast and this was directly influencing the behavior of the loss curve, something i could see with this technique and the plots. it was like seeing a ghost in the machine, but i managed to exorcise it, eventually.

i recommend you check out "deep learning for vision systems" by mohamed elgendy, he explains training dynamics really well and what to look for and also the "deep learning with pytorch" book from eli stevens, luca antiga, and thomas viehmann. they are really good resources and give you some good insights into deep learning concepts. the official fastai documentation is always a good resource to check out as well but i'm assuming you've already checked that.

one last thing, and it’s not entirely related but i always tell this to my younger colleagues to always keep in mind, one time i tried to train a model without normalizing the data first. i was wondering why the losses were not decreasing properly... i was scratching my head for hours. i had a major "facepalm" moment afterwards.

i hope this detailed explanation helps you get what you need. let me know if anything is unclear, always happy to help.
