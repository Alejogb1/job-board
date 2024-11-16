---
title: "Deploying AI: From Demo to Production"
date: "2024-11-16"
id: "deploying-ai-from-demo-to-production"
---

dude so this talk was all about getting ai stuff outta the demo phase and into actual use like real world stuff  it was wild  the dude started by asking how many peeps had llms already running in their companies  and like whoa seventy percent plus  it was bananas  most folks already had this figured out  but they were clearly still struggling  which is why they were at the talk  right

the whole setup was kinda hilarious  he was using a slideshow but there was some initial tech drama  like he almost showed his ip address to the whole audience  which is a major no-no  classic  then he did this quick poll thing about whether companies were building their own llm solutions or buying pre-made ones  custom solutions were way more popular  like thirty percent vs way less for purchased solutions  he even threw github copilot into the purchased category which i thought was kinda funny

one of the big ideas was how easy it is to make an ai demo that looks amazing but then super hard to actually turn it into something useful in production he said  something like 'ai makes ceos stupid' which i thought was a pretty bold statement but totally accurate  haha  he mentioned he gets these incredible looking demos all the time  but they're miles away from being ready for prime time  it's because ai is so experimental  you're constantly tweaking and trying new things  unlike regular software development which is much more linear

another key concept was the importance of tracking everything  because with ai the learning process is your intellectual property not the model itself  he said it like this  "if you don't save that when the person that figured out walks out the door ip walks at the door with them"  super true  if you don’t track all your experiments you lose everything   it's not just about protecting your secrets  it's about making your work reproducible so others can build on it which makes the whole process way faster  you know collaboration is key for speeding things up  which is essential to get from demo to production fast

he gave this awesome real-world example  he built an alexa-like thing for his daughter  after his kid was born lol  his daughter kept asking alexa to play her favorite song which was baby shark  but alexa didn't know because alexa is apparently not great at tracking repeated requests  so he decided to build her a custom one using open source stuff  like llama2  and a raspberry pi  which is awesomely low-key tech

the resolution of the story is that building it was a journey  he had to do a ton of experimenting  prompt engineering  model switching   and finally fine-tuning using a technique called loRA  to get it up to 98 percent accuracy  which honestly is insane  the process itself is a perfect example of what he was talking about  he used lots of different techniques along the way  and it took multiple iterations to get it working

now for the code  since i'm your friendly neighborhood code-slinging buddy  here are a few snippets  all inspired by what he talked about

first prompt engineering  a simple example of improving a prompt to get better results from llama2


```python
# bad prompt
prompt = "what's the weather"

# improved prompt  more specific  and formatted for better response
prompt = """
provide the weather in Boston in the following format:

location: <location>
temperature: <temperature>
conditions: <conditions>
"""
```

see the difference  the second prompt is way more specific and directs the model toward the desired output format


next is a super basic example of how you might structure an api call based on an llm's output


```python
import requests

# llm output  assume it's perfect for now
llm_output = "weather.get_weather(location='boston')"

# extract location from the llm output using simple regex  in real life you'd have more error handling
import re
match = re.search(r"weather\.get_weather\(location='(.*?)'\)", llm_output)
location = match.group(1) if match else None

# api call  replace with your actual api key and endpoint
api_url = f"https://api.example.com/weather?location={location}&api_key=YOUR_API_KEY"
response = requests.get(api_url)

# process the response
weather_data = response.json()
print(weather_data)
```

this shows  the llm generating a function call  which then gets parsed and used to make an actual request


finally a tiny snippet showcasing the idea of fine-tuning using a simple dataset format


```python
# example data points for fine tuning the llm
data = [
    {"input": "what's the weather in london", "output": "weather.get_weather(location='london')"},
    {"input": "what's the temperature in new york", "output": "weather.get_temperature(location='new york')"},
    {"input": "tell me about the weather in paris", "output": "weather.get_weather(location='paris')"}
]
```


this is super basic  but it shows a simple format for fine-tuning data  input and output pairs  which is essential for teaching your model to generate specific output



so yeah  that talk was a wild ride  lots of laughs and some really insightful points about building and deploying ai  the key takeaways were the huge gap between demos and production  the importance of tracking everything  building good evaluation frameworks  and iterating  iterating  iterating  it’s all about the process  and having fun while doing it
