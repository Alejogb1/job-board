---
title: "Deploying Open-Source AI Models for Production"
date: "2024-11-16"
id: "deploying-open-source-ai-models-for-production"
---

dude so this dima guy gave this killer talk about fireworks ai and lemme tell ya it was a wild ride  basically the whole point was showing how they make using open-source ai models super easy even for production-level stuff which is usually a total nightmare  think less wrestling with gpus and more actually building cool stuff

first off dima sets the stage talking about his team’s mad skills  they’re like ex-meta and google ai peeps  so they've seen it all  he even mentions being a core maintainer of pytorch for five years  that's some serious cred  the visual cue here was him giving a little humblebrag about his background  you could practically see the years of late nights debugging flashing across his face


then he hits us with the core problem:  big models like gpt-4 are awesome but they're expensive slow and overkill for lots of tasks  he used a great visual aid a slide showing the cost comparison  it was pretty brutal  like deploying a giant model can cost you millions yearly but a smaller tuned model?  way cheaper  his point is crystal clear you often don’t need a language model that can write sonnets and identify pokemon to answer simple support queries

he lays out three key challenges in using open-source models for production  first setup and maintenance is a beast  imagine wrestling with different gpu setups and framework versions every time a new model update drops  second optimizing these things is a black art  there are so many parameters to tweak and it’s not obvious which settings give you the best speed and quality for your specific use case  third getting everything production-ready is a whole other ball game  you need scalability monitoring and other enterprise-grade features which are no joke

now for the juicy stuff  fireworks ai's solution they've built their own crazy-fast serving stack which is allegedly the fastest in the biz  they wrote custom cuda kernels which means they tweaked the core gpu instructions to maximize performance  it's like they hand-tuned every tiny detail  the visual here was a benchmark showing fireworks blowing away the competition on long prompts  they specifically optimized for those long prompts because using a knowledge base aka rag often requires them  a key concept here is that they're not just serving models they're deeply optimizing the whole serving infrastructure for your needs

here's a tiny taste of what that optimization looks like  this is a super simplified version just to give you a flavor  imagine you want to optimize the model for a specific latency target


```python
import torch

# hypothetical model loading and preprocessing
model = torch.load("my_awesome_model.pt")
preprocess = lambda x:  #your preprocessing function here

# latency constrained optimization loop

target_latency = 0.2 # seconds
best_config = {}
min_latency = float('inf')

configs = [
    {'batch_size': 1, 'precision': 'fp16', 'quantization': False},
    {'batch_size': 4, 'precision': 'fp32', 'quantization': False},
    {'batch_size': 8, 'precision': 'int8', 'quantization': True}, # example of quantization
    # ... more configs
]

for config in configs:
    # hypothetical latency measurement (replace with your actual measurement)
    latency = measure_latency(model, preprocess, config)

    if latency < min_latency and latency <= target_latency:
      min_latency = latency
      best_config = config
      print(f"Found better config: {best_config} with latency {latency}")
print(f"Best config found: {best_config}")

# ... rest of inference loop using best config
```

this code snippet shows a simplified way to experiment with different configurations  in reality you need a robust way to measure latency maybe using a tool like  `perf` or something similar  and you'd have more sophisticated configurations including things like different gpu scheduling strategies and memory allocation settings


another key idea is their focus on  customization  they don't just serve you a model they let you tweak it to your heart's content  they talked about how they handle fine-tuned models  imagine you fine-tune a llama model for a specific task  they can deploy tons of these variations on the same gpu so you only pay for what you use its a killer feature  they’re not selling you a box of tools they're offering a whole ecosystem

here’s a snippet showing a simplified idea behind model selection based on latency/cost tradeoffs

```python
import pandas as pd

models = pd.DataFrame({
    'model_name': ['llama-7b', 'llama-13b', 'llama-30b'],
    'latency': [0.1, 0.3, 1.0],
    'cost_per_token': [0.0001, 0.0003, 0.001],
    #... other metrics
})

# define a cost function that incorporates latency
def cost_function(row, max_latency):
  if row['latency'] > max_latency:
    return float('inf') #reject this model since it exceeds max latency
  else:
    return row['cost_per_token'] * row['latency'] #this could be improved to account for more factors


max_allowed_latency = 0.2 # example max latency

models['total_cost'] = models.apply(cost_function, args = (max_allowed_latency,), axis = 1)

best_model = models.loc[models['total_cost'].idxmin()] #find the model with the minimum cost given our latency constraint

print(f"Best Model: {best_model['model_name']}")

```

this is a  simplified  example showing how to select a model based on minimizing cost subject to latency  you would use a more realistic cost model including factors like gpu hours and other infrastructure expenses  plus a much more comprehensive model database


finally they talk about building  compound ai systems which is the current buzzword  it's not just about the model itself but the whole system  it involves connecting your language model to other things like knowledge bases databases and apis using the magic of function calling which is like giving the model superpowers

here's a simple function-calling example using a hypothetical library


```python
from hypothetical_function_calling_library import FunctionCaller

# initialize the function caller
fc = FunctionCaller(model, api_keys={"stock_api": "your_api_key"})

user_query = "show me the top 3 cloud providers' stock prices"

response = fc.call(user_query, functions=["get_stock_prices"])

print(response)
```

this would interact with a hypothetical api  the actual implementation would be far more complex involving things like prompt engineering to instruct the model how to call external functions plus sophisticated error handling for these external calls


the ending?  fireworks ai wants you to try their stuff  they’re open api compatible so you can hook it into whatever you already use  they've got a playground for quick tests and they handle everything from tiny experiments to massive production deployments and that’s the story of fireworks ai from the perspective of a friend who just happened to watch the talk and was blown away  pretty neat right?
