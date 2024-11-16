---
title: "LLM Speed and the Future of Computing"
date: "2024-11-16"
id: "llm-speed-and-the-future-of-computing"
---

dude so this video was like a total mind-melt about how llms are gonna freakin' explode the tech world faster than a processor hitting 1ghz in 1999  it's all about the insane speed improvements we're seeing and how that's gonna totally reshape everything

the whole point was to show how llm speed is ramping up way faster than even moore's law predicted  think about it  we went from practically nothing to gigahertz processors in like two decades  now llms are already showing even more insane progress

some visual cues  i remember the speaker showed this killer press release announcing intel's 1ghz chip  that was seriously a blast from the past  then he showed a chart about llm speed increases it was steep as heck  plus there was a screenshot of globe.engineer a service that plans trips insanely fast using an llm seriously awesome

ok so two key concepts  one was this idea of llms becoming the next big thing in computing remember how we went from paper processes to digital ones to connected ones and now to these mobile form factor changes  this dude said now we're at the industrialization phase for tech  meaning llms will start to automate and scale things in a way that's never happened before  like imagine a world where all the boring and repetitive tasks are handled seamlessly by an llm

the other was the speed implications like they're talking 10000 tokens per second or some crazy number that let's you plan a trip to new york with pizza recommendations in like 5 seconds  no joke  think about how many tabs you'd open to do that yourself insane  he also talked about this move towards instantaneous responses and  multimodal interactions that stuff is legit wild

code time  this is where it gets fun

first imagine building a simple llm-powered trip planner  something like this

```python
import openai

def plan_trip(destination, interests):
  prompt = f"plan a trip to {destination} focusing on {interests} including flights hotels and food recommendations"
  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=200
  )
  return response.choices[0].text.strip()


trip_plan = plan_trip("new york", "pizza and sightseeing")
print(trip_plan)
```

so you see  we're using the openai api  that's the heart of many llm apps  this tiny snippet gives you a basic idea you'd need a lot more code to handle booking flights hotels etc  but you get the idea  instant trip planning

next let's think about  a super basic example of a multishot llm interaction you could expand on this enormously

```python
import openai

def multishot_query(initial_prompt, follow_up_prompts):
  full_prompt = initial_prompt
  for prompt in follow_up_prompts:
    full_prompt += "\n" + prompt

  response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=full_prompt,
    max_tokens=200
  )
  return response.choices[0].text.strip()

initial = "write a story about a talking dog named bob"
followups = ["bob meets a cat", "the cat is a secret agent"]

story = multishot_query(initial, followups)
print(story)

```
this is where you give the llm a story prompt then more prompts  to refine and build on it that's one simple way to show how multi-shot queries work to get better results than a single shot one

finally consider a super simplified example of context-aware processing  imagine it as enhancing a chatbot  

```python
# highly simplified example, needs tons of robust error handling and real context management
user_context = {"name": "john", "location": "london", "preferences": ["coffee", "museums"]}

def contextual_response(user_input):
    prompt = f"user: {user_input}\ncontext: {user_context}\nrespond:"
    # (in a real app, you'd use a much more sophisticated llm call here)
    response = "based on your preferences in london, i suggest visiting the british museum and grabbing a flat white"
    return response

print(contextual_response("what should i do today"))
```

this is ridiculously oversimplified but you get the point we store user info and feed it to the llm which then provides a tailored response  that's the essence of context awareness  imagine that with tons of data  its implications are huge

the conclusion was pretty straightforward  the speed of llms is growing ridiculously fast  and it's going to cause a total paradigm shift in how we build and use software  think self-driving cars meeting industrial revolution levels of change  it's a huge deal  the video really made me think about how llms will become the core of many computing systems not just an add-on  that's a pretty crazy thought and it's all happening way faster than anyone expected
