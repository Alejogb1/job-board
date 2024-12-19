---
title: "AI System Evaluation:  Zapier & Brain Trust Approach"
date: "2024-11-16"
id: "ai-system-evaluation--zapier--brain-trust-approach"
---

yo dude so i just watched this totally rad talk about how zapier and brain trust are using ai like total ninjas  it's all about building and testing their ai stuff  and honestly it blew my mind  they're not just building cool things they're also figuring out how to make sure those things don't totally explode in their face which is like super important  

the whole thing kicks off with these two dudes malon and anker  they're like the chillest engineers ever malon's this super relatable guy he's like "yeah i've written some bugs you might have run into sorry about that"  anker's a little more focused on the big picture the whole brain trust platform they're using for all their ai testing stuff it's like their secret weapon  

one of the first things i caught was anker mentioning  "over 7000 apps"  that's zapier's insane integration count  that's a crazy amount of stuff to keep track of which makes their testing process even more crucial  later on they show a screenshot of their internal prioritization the p0 stuff was totally clear "triggers actions working perfectly top 25 apps supported"  that's smart because you gotta nail the core stuff before messing around with niche stuff  

then we dive into the nuts and bolts of their evaluation system  they're not just doing some half-assed testing oh no  they've built this epic framework for evaluating their ai using brain trust which is a platform specifically designed to handle the crazy complexities of evaluating llms and ai systems it's like a supercharged testing environment  this whole setup is super impressive it uses synthetic data for testing running on a ci/cd pipeline  they mentioned 800 tests moving from a measly 7 manual unit tests to that many automated ones is a huge leap  

seriously dude their eval system is like a masterpiece  they're using custom graders  both logic-based and llm-based  to make sure their ai is performing exactly as it should be  they're not just looking at pass or fail they're measuring all sorts of things like accuracy speed  and even stuff like whether the ai is following their instructions properly and handling edge cases  and get this they're running these evals super frequently  the whole thing is designed for continuous monitoring so they can quickly spot and fix any problems  that's what separates the pros from the amateurs  

and get this  the way they use brain trust for observability is genius   they can see everything that's going on inside their ai systems the inputs the outputs the timing  even the tokens  it's like having x-ray vision for your ai   this is why their setup is so great they're not just fixing bugs they're learning *why* the bugs happen  it's proactive development at its finest  

here's a tiny code snippet to give you a taste  this is simplified but it's like how they might track the performance of different models  

```python
import time

models = {
    "gpt-3.5-turbo": [],
    "gpt-4": []
}

for model_name in models:
    start_time = time.time()
    # Simulate an API call to the model
    response = f"response from {model_name}" 
    end_time = time.time()
    models[model_name].append(end_time - start_time)

for model_name, times in models.items():
    avg_time = sum(times) / len(times) if times else 0
    print(f"{model_name}: average response time = {avg_time:.4f} seconds")

```

that's a super basic example but it shows the kind of data they're collecting  they're tracking everything so they can easily see how different model choices or parameter changes affect performance  this is all feeding into their iterative approach which is seriously awesome  

another key idea they pushed was involving product managers in the loop  it's not just an engineering problem  it's a product problem too  they're making sure the ai works the way the users expect it to  this collaborative approach is key to building something useful and user-friendly  

they also show this awesome graph that shows how their evaluation scores changed over time after they switched to gpt-4  their scores initially dropped  like way down  it was a huge regression but through careful analysis and some clever prompt engineering they managed to get their scores back up  it's a great example of how even the most sophisticated systems require careful monitoring and refinement  

here's a little more code representing the kind of prompt engineering they did  they started with a really detailed prompt and then simplified it   

```python
# initial overly specific prompt
initial_prompt = """
create a zapier zap to do the following very precisely:
trigger: when a new row is added to google sheets with specific column names
action: update airtable base with matching data
ensure the field mapping is exact
do not use other applications besides google sheets and airtable
"""

#refined prompt
refined_prompt = """
create a zapier zap:
trigger: new google sheets row
action: update airtable
"""
```

see the difference  the initial prompt is overly restrictive  the second one is more flexible and lets the llm do its thing more efficiently  

and another snippet showing some basic performance tracking using a dictionary for simplicity

```python
performance_data = {
    "gpt-3.5-turbo": {
        "accuracy": 0.95,
        "speed": 10  #seconds
    },
    "gpt-4": {
        "accuracy": 0.98,
        "speed": 20 #seconds
    }
}

for model_name, data in performance_data.items():
  print(f"{model_name}: accuracy={data['accuracy']}, speed={data['speed']} seconds")
```

this type of simple data structure is useful for quick comparisons and helps them make data-driven decisions about which model to use   

the whole talk ends with them showing their new chat-based interface  it's a massive upgrade they're making it even easier for users to interact with their ai  and of course they're still using brain trust to track everything  to maintain their high level of quality and provide a consistent and efficient experience  

so yeah dude that's the lowdown on the zapier brain trust collab  it's a masterclass in building and evaluating ai  the biggest takeaway is the importance of continuous evaluation  iterative development and a deep focus on observability  they're not just building cool ai they're building a system for building and improving even cooler ai  it's all about that continuous improvement loop  and they're doing it with style and humor  pretty epic right
