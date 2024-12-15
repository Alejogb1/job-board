---
title: "How to get multiple plots from multiple learners on one plot?"
date: "2024-12-15"
id: "how-to-get-multiple-plots-from-multiple-learners-on-one-plot"
---

alright, so you're looking to cram several learning model outputs onto a single visualization. been there, done that, got the t-shirt (and probably a few gray hairs). it's a common need when you're comparing how different models are performing, or maybe you're showcasing how they converge over training epochs.

first thing, let's establish what we mean by 'plots'. i'm assuming you're working with some kind of numerical data, whether it's regression outputs, classification probabilities, or even loss curves. also, i'm figuring you have multiple learners – think different algorithms like linear regression, support vector machines, decision trees, or even neural nets.

the general idea boils down to this: you need a plotting library that can handle multiple series on one axes, and you need to structure your data so it plays nicely with it. i've personally bounced between matplotlib, seaborn, and plotly depending on the project needs. each one has its quirks and strengths, so it's good to have a few in your toolkit.

let me tell you about a time i messed this up real bad. it was back in my early days, building a sentiment analysis model. i trained a naive bayes and a recurrent neural network on the same dataset, and wanted to visualize their performance on some held-out data. instead of overlaying the predictions on one plot, i generated two separate plots, using default axis ranges. needless to say, the visualizations were practically useless for comparison since the scales were all different. it was like comparing apples to oranges, and i spent like half an hour looking at two different plots like they were supposed to tell me something. lesson learned: always ensure your x and y axis scales make sense, or you will spend too much time thinking about the wrong thing.

anyway, let's get down to some practical examples. i'm going to demonstrate with matplotlib, since it's kind of the workhorse for python plotting, but the same principles can be applied to other libraries.

here's the simplest case: we have two lists of values, one for each learner, and we want to plot them against a common x-axis (let's assume they share the same data points).

```python
import matplotlib.pyplot as plt
import numpy as np

# fake data for two learners
x = np.linspace(0, 10, 100)
learner1_output = np.sin(x)
learner2_output = np.cos(x)

# create the plot
plt.plot(x, learner1_output, label='learner 1')
plt.plot(x, learner2_output, label='learner 2')

# add some bells and whistles
plt.xlabel('x-axis data points')
plt.ylabel('model output values')
plt.title('comparison of two learner outputs')
plt.legend()  # show the labels we used
plt.grid(True)  # for easier reading
plt.show()

```

this code does the basic job: it plots two lines, one for each learner's output, on the same plot. the `label` argument is important, because it enables you to add a legend so you know which line corresponds to which model. the `grid(true)` makes the visualization less of a mess.

now, let's say your learner outputs are actually prediction probabilities, and you want to visualize them using a scatter plot. this comes up a lot when you're doing classification. here's how you could do it:

```python
import matplotlib.pyplot as plt
import numpy as np

# more fake data, this time for a classifier
x = np.random.rand(100)
learner1_probabilities = np.random.rand(100)
learner2_probabilities = np.random.rand(100)

# create scatter plot
plt.scatter(x, learner1_probabilities, label='learner 1 probabilities', marker='.', alpha=0.5)
plt.scatter(x, learner2_probabilities, label='learner 2 probabilities', marker='x', alpha=0.5)

# the jazz
plt.xlabel('input feature space')
plt.ylabel('prediction probabilities')
plt.title('classifier output probabilities')
plt.legend()
plt.grid(True)
plt.show()
```

notice that i'm using `marker='.'` and `marker='x'` to distinguish between the two models in the scatter plot, and that transparency added through alpha helps the reader, this helps a lot when you have many overlapping points. it's good to think about how you want the markers to help the audience to see what’s going on, or if markers are the correct way to show the data.

let's level up a bit. imagine you want to compare the loss curves of multiple neural networks as they train. this means plotting multiple lines where the x-axis is the training epoch. the key here is that your training loop needs to keep track of the losses for each epoch and each model.

```python
import matplotlib.pyplot as plt
import numpy as np

# faking training loss for two models
epochs = np.arange(100)
model1_losses = np.exp(-epochs / 20) + np.random.normal(0, 0.01, 100)
model2_losses = np.exp(-epochs / 15) + np.random.normal(0, 0.02, 100)

# create loss curve plot
plt.plot(epochs, model1_losses, label='model 1 loss curve')
plt.plot(epochs, model2_losses, label='model 2 loss curve')

# the usual
plt.xlabel('training epochs')
plt.ylabel('training loss')
plt.title('comparing training curves')
plt.legend()
plt.grid(True)
plt.show()
```

the data used for the curves are synthetic, so take it with a grain of salt. one time i spent two hours looking at a loss curve that wasn't decreasing because of a bug in my data loader. fun times… not really.

now, before i sign off i recommend that if you want to really dive deep on visualization you must check out 'interactive data visualization for the web' by scott murray. if you need a good resource on statistical graphics, check out 'the visual display of quantitative information' by edward tufte. these are old, but trust me, their principles are timeless.

a few final points to keep in mind:

*   always label your axes and add a title. don't make the reader guess what they're looking at.
*   use a legend to clarify which line or point corresponds to which model.
*   choose the right plot type for the data. are lines, scatterplots, bar charts, or heat maps more appropriate? that's always a tricky question.
*   think about color schemes and the overall aesthetic. a little bit of thought makes a big difference on how easy your plots are to read.
*  if you have many models or high dimensionality consider alternative ways to summarize the model differences. a few metrics may give you more insight than trying to plot 100 models in the same graph (i know someone who did that once, not recommending it).

that should get you started. plotting multiple learners on a single plot isn't that tricky once you get the basics down. the hardest part is usually deciding what to show, or not to show.
