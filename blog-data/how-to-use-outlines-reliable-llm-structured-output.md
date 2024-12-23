---
title: "How to Use Outlines: Reliable LLM Structured Output"
date: "2024-11-16"
id: "how-to-use-outlines-reliable-llm-structured-output"
---

yo dude so i just watched this awesome talk about outlines this library for making llms way less flaky and way more useful it’s all about structured output and it blew my mind lemme break it down for ya

the whole point of the talk was basically this llms are kinda dumb they spit out words but they don’t really *get* the structure of what they're saying  remy the guy presenting  showed how you ask an llm for flight info from an email and it just throws a json decode error at you  like wtf right  the whole point of computing is that things have consistent apis you can rely on but llms are all over the place  so outlines is the solution – it lets you tell the llm exactly what structure you want the answer to have so you get reliable results every time  it's like adding a brain implant to the dumb llm

 so five key moments that totally stuck with me

1. **the json decode error problem:** this was the killer opening  remy hammered home how frustrating it is when an llm promises structured data but delivers gibberish instead it's the biggest reason why we can't trust them to do serious stuff easily like taking data and putting it into a database
2. **the power of structured generation:** this is where outlines shines it's not just about generating text it’s about guiding the llm to generate text *in a specific format*  like json or by using regex this turns unreliable word salad into neatly packaged data points  imagine you need a list of email addresses from a thousand emails  forget regex hell just use outlines and tell it to give you a json array of those addresses – simple clean and efficient
3. **how outlines works:**  it’s a surprisingly straightforward idea during text generation the llm produces probabilities for the next word outlines intercepts those probabilities and nukes any word that would violate the desired structure  it's like a bouncer at a club only letting words in that fit the theme  the clever part is that they’ve made this super fast it's mostly transparent to the user which is awesome
4. **the speed boost:** this totally blew me away  using structured generation not only gets you clean structured outputs it actually *speeds up* the whole process  why because the llm isn’t wasting time generating irrelevant tokens or “yapping”  remy showed how much fewer tokens you need to generate when you know exactly what structure you’re aiming for it’s like writing a haiku instead of a rambling novel for the same info  efficiency is king in machine learning and outlines totally delivers
5. **open source models are the future:** this is where remy gets evangelical  he’s super excited about the potential of open-source models combined with structured generation they can compete with and even outperform closed-source giants like openai  the fact that a smaller model with structure generation can achieve better accuracy and speed than a huge model without it is wild  the democratization of AI is a huge deal.

there were some really cool visuals too the memes were great especially the one about the chaotic attempts to get valid json from an llm  and then the graphs showing the huge accuracy jump and almost zero overhead with outlines versus other methods – those were very persuasive

  let's get to the nitty-gritty  two key concepts  regex and json schema

**regular expressions (regex):**  think of regex as a pattern-matching superpower  it lets you define patterns of characters that you want to search for within text   for example  `\b[A-Z]{3}\b`  would find any three-capitalized-letter words –perfect for airport codes  in outlines you give it the regex and it tells the llm "only generate text that matches this pattern"  no more sifting through endless text to extract your information  elegant

```python
import re

text = "the flight departs from LAX and arrives at JFK"
pattern = r"\b[A-Z]{3}\b" # regex pattern for 3-letter uppercase words
matches = re.findall(pattern, text)
print(matches) # Output: ['LAX', 'JFK']
```

this simple code finds the airport codes from the text using regex  outlines uses this idea  but instead of searching text it tells the llm to generate text that matches the regex in the first place – far easier and more efficient

**json schema:** json is basically a universal way to represent data in a structured format – like a database table  a json schema defines the *shape* of a json object which fields it has what data types they are  for example you could define a schema for a flight  with fields like "origin" "destination" and "flight number"  outlines uses this schema to tell the llm exactly what kind of json to generate   guaranteeing that your output will be well-formed and usable

```python
import json
from jsonschema import validate

schema = {
    "type": "object",
    "properties": {
        "origin": {"type": "string"},
        "destination": {"type": "string"},
        "flightNumber": {"type": "integer"}
    },
    "required": ["origin", "destination", "flightNumber"]
}

data = {"origin": "LAX", "destination": "JFK", "flightNumber": 1234}

try:
    validate(instance=data, schema=schema)
    print("valid json")
except Exception as e:
    print(f"invalid json: {e}")

```

this code checks if a json object matches the specified schema  if not it throws an error – outlines uses this to make sure the llm’s json output is perfect


here's some outlines code to sweeten the deal

```python
from outlines import Generator, models

generator = Generator(model=models.llama2) #choose your model

#generating a single sentence
response = generator.text(
    prompt="describe the benefits of structure generation in one sentence",
    stop=["."],
)

print(response)

# using a regex to guide structure
response = generator.regex(
    prompt="what is the ip address of the public google dns servers",
    regex=r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
)

print(response)

# json schema example

schema = {
  "type": "object",
  "properties": {
    "flight": {
      "type": "object",
      "properties": {
        "origin": {"type": "string"},
        "destination": {"type": "string"}
      },
      "required": ["origin", "destination"]
    }
  },
  "required": ["flight"]
}

response = generator.json(
  prompt="extract flight info from this email blah blah blah LAX JFK blah",
  schema=schema,
)
print(response)

```

so in short the talk's conclusion was that structured generation using outlines is a game changer for anyone not building chatbots  it makes llms reliable efficient and way more useful  the speed boost the accuracy improvements the guaranteed valid output it’s a total win. remy really made a compelling case for this technology and i’m definitely adding outlines to my toolbox  it’s open source too so go check it out  you won’t regret it
