---
title: "How does Google’s Project Mariner achieve state-of-the-art performance on the WebVoyager benchmark?"
date: "2024-12-12"
id: "how-does-googles-project-mariner-achieve-state-of-the-art-performance-on-the-webvoyager-benchmark"
---

Alright let's dive into Project Mariner and how it conquers WebVoyager its a pretty cool area to geek out on actually

So WebVoyager think of it as this intricate obstacle course for web-based agents Its not just about clicking buttons randomly its about understanding the web page layout its dynamic content and figuring out the user intention essentially the agent needs to be able to navigate different websites accomplish specific tasks like booking flights or ordering takeout this is where traditional rule based systems often stumble because the web is so diverse and always changing

Project Mariner its not like your average search bot it leverages a whole bunch of sophisticated deep learning techniques its about creating an agent that learns to perceive the web as a human does or at least a very good mimic of it a key aspect is the use of large pre-trained transformer models models like the ones behind GPTs or BERT but customized for web interaction these transformers have seen a ton of text and code giving them a really solid grasp on language and structure

They are fed raw HTML that ugly markup that makes up the web page structure but this isn't just like dumping a pile of text on the model they embed it transforming the HTML into a format that the model can actually understand and not only the html but also information such as images and layout is included in this embedding allowing the model to perceive the visual presentation of the webpage

Then comes the tricky part actually interacting with the website the agent needs to figure out what actions to take what elements to click on what text to enter this requires reasoning and planning which is also done using transformer based models with a reinforcement learning approach The agent starts by trying random actions and over time it learns which actions are successful in completing the task this is very similar to how you would train a self driving car

The cool thing about the Mariner architecture its end-to-end meaning the models learns to do this all together from the web page understanding to the task completion from raw input to target action with only very little human intervention its not like a bunch of hand coded rules or heuristics which make it really adaptable and flexible to new websites

Now you asked about state-of-the-art results on the WebVoyager benchmark the reason it does so well is basically threefold

First the massive pre-trained models these giants bring a huge amount of general knowledge about language and the web which other agents often lack It means that the model doesnt need to learn from zero for each new website it can understand the layout the content and the way websites function right out of the gate.

Second the end-to-end training this approach allows the models to directly optimize for the final task in hand such as completing the booking or the purchase action instead of optimizing a number of intermediate tasks and then piecing them together this really enables better performance and allows the agent to learn more complex behavior.

Third the use of reinforcement learning enables the agent to continuously learn and improve from experience the model is not programmed to do a set of tasks but learns through trial and error it is an ongoing learning process so it can tackle challenges it has never seen before.

Lets look at a simplified example of how HTML is turned into a more model friendly format using some python code:

```python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

html_string = "<div class='item'> <p>This is an example item</p> <button>Click me</button> </div>"

tokens = tokenizer.tokenize(html_string)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids_tensor = torch.tensor([token_ids])

with torch.no_grad():
    outputs = model(token_ids_tensor)
    embeddings = outputs.last_hidden_state
print(embeddings.shape) # This outputs the dimensions of the output
```
This snippet shows how to take basic html represented as a string tokenize it and feed it to a pretrained model This resulting tensor represents the webpage in a format the model can work with

And here is an example of how the action selection process could be simplified using an imaginary action space with some basic elements and some probabilities:

```python
import torch
import random

# Example actions
actions = ["click button", "enter text", "scroll down", "hover element"]

# Let's imagine the model outputs some probabilities for each action
action_probabilities = torch.tensor([0.4, 0.2, 0.3, 0.1])

# The probability with the highest value is selected
selected_action_index = torch.argmax(action_probabilities)
selected_action = actions[selected_action_index]

print(f"Selected action: {selected_action}")

# We can also implement a method where we select the action based on the probabilities
# instead of just taking the max probablity this can improve exploration of the system
def probabilistic_action_selection(action_probabilities):
  random_choice = random.uniform(0,1)
  cumulative_probability=0
  for i, prob in enumerate(action_probabilities):
    cumulative_probability +=prob
    if random_choice < cumulative_probability:
      return i
  return len(action_probabilities)-1 # handle potential rounding errors

selected_action_index_probabilistic = probabilistic_action_selection(action_probabilities)
selected_action_probabilistic = actions[selected_action_index_probabilistic]

print(f"Probabilistically Selected action: {selected_action_probabilistic}")
```

This code example shows how the action selection process in reinforcement learning could look like and how this is used to drive the web agent

Finally here is a way to simplify the reward function:

```python
def reward_function(current_state,target_state,is_complete):
    if is_complete:
        return 10
    distance = calculate_distance(current_state, target_state)
    return -distance * 0.1

def calculate_distance(current_state,target_state):
    # we would have to define how we calculate the distance between these two states in
    # an actual scenario this example is purely abstract
    return abs(current_state - target_state)

current_state = 10
target_state = 20
is_complete = False
reward = reward_function(current_state, target_state, is_complete)
print(f"Reward: {reward}")

current_state = 20
target_state = 20
is_complete = True
reward = reward_function(current_state, target_state, is_complete)
print(f"Reward if goal is achieved: {reward}")
```

This shows how an agent can be given rewards when it is progressing towards its goal and a larger reward when the goal is finally achieved this reward is what guides the system through reinforcement learning.

For resources to get deeper into this stuff you should check out the original papers on transformer models like "Attention is All You Need" you can find them on arXiv its a must read And for the reinforcement learning side of things "Reinforcement Learning an Introduction" by Sutton and Barto its the bible in that field. There are also tons of open source deep learning tutorials using PyTorch and TensorFlow that could help you understand the code.

This whole area of web agents and how they can navigate the web is really exciting it's all about making computers better understand and interact with us making tasks more easier and efficient Its a big leap from the traditional rule-based systems and Mariner is at the forefront.
