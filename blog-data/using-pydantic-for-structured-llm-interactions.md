---
title: "Using Pydantic for Structured LLM Interactions"
date: "2024-11-16"
id: "using-pydantic-for-structured-llm-interactions"
---

yo dude so this keynote was *wild*  the guy was talking about type hints and how this thing called pydantic basically makes working with language models way less of a headache than it usually is  he was saying like 90% of the time when you use these LLMs you're just trying to get some structured data back like json and parsing that garbage is a nightmare you're basically wrestling a greased pig in a phone booth  the whole point of the talk was showing how pydantic and a few other libraries can make that whole process way smoother


first he totally called out how messed up it is to just pray to the LLM gods that your json comes out perfect  i've been there  you spend hours tweaking the prompt just to get the right commas and brackets and then the keys are all slightly different from one call to the next  he showed this hilarious tweet from riley goodside where the only way he got bard to spit out proper json was by threatening to end the world which is not exactly maintainable code lol


one key idea was openai's function calling feature which helps a lot you define a json schema of what you expect back and openai tries harder to give you that but even then you still gotta parse it using `json.loads()` which is a whole other potential minefield he was like "you're still praying"  


then he introduced pydantic this library is *amazing* it does data validation like a boss based on type hints and it generates json schemas which are exactly what you need to talk to the openai function calling thing  he said it has 70 million downloads a month which is insane it's clearly something people trust



here's a little code snippet to show how it works

```python
from pydantic import BaseModel, Field
from datetime import datetime

class Delivery(BaseModel):
    timestamp: datetime = Field(..., description="Delivery timestamp")
    dimensions: tuple[float, float, float] = Field(..., description="Dimensions of package (length, width, height)")

delivery = Delivery(timestamp="2024-10-27 10:30:00", dimensions=[10.5, 5.2, 2.1])  # string and list are automatically converted
print(delivery.json()) # json representation 
print(delivery.timestamp)  #easy access to typed data

delivery_fail = Delivery(timestamp="this is not a datetime", dimensions="wrong type")  # throws a validation error
```


see how clean that is you define your data structure using pydantic and it handles all the parsing and validation automatically  you get type checking autocompletion and everything  its *magical*


he also talked about a library he built called instructor which basically makes using pydantic with openai super easy  it patches the openai api so your function calls automatically use your pydantic models as the response type it's all about type safety so your IDE can understand what's going on


```python
import openai
from instructor import instruct
from my_pydantic_models import MyModel


@instruct(response_model=MyModel)
async def get_data(prompt: str):
    response = await openai.Completion.acreate(model="gpt-3.5-turbo-0613", messages=[{"role": "user", "content": prompt}])
    return response #automatically converted into MyModel if possible

result = await get_data("give me some structured data")
print(result.some_field)
```

this snippet shows how easy it is to get the type safety from pydantic all the way into your openai integration  before i'd write functions that would return dictionaries then spend hours combing through them to pull out specific pieces of information and it would break randomly


another awesome bit was using pydantic for more than just LLM stuff  he showed how you could create custom validators


```python
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str

    @validator("name")
    def lowercase_name(cls, value):
        if isinstance(value, str):
            return value.lower()
        else:
            raise ValueError("name must be a string")


user = User(name="Bob")
print(user.name)  # Output: bob

user_fail = User(name=123)
#this will raise a ValueError as the name isn't a string.
```


he even used an LLM as a validator to check if a string was "mean" which is just mental  like you define rules for your data and the LLM enforces them  


the resolution was that structured prompting using pydantic and similar libraries is the way to go  it's all about making your interactions with LLMs cleaner more predictable and easier to maintain  it's not just about better json its about getting structured outputs that actually reflect *your* structured thoughts and workflow the speaker mentioned a few other libs like marvin  that do similar things but add more features



he also showed some more advanced examples like building rag systems and query planners using this approach  instead of writing messy prompt chains you can define the whole workflow as a data structure which is brilliant he even showed how to extract knowledge graphs and do question answering in a way that prevents hallucinations by verifying quotes against the source text  the code was much cleaner and easier to follow because of the type safety and structured approach


overall the talk was incredibly insightful  it highlighted how type hints and structured data can make working with LLMs a thousand times better its not just about making your code work its about making your code *understandable* *maintainable* and  *less stressful* to work with  i'm totally going to start using pydantic for all my LLM projects now i felt like i left the talk having leveled up as a programmer and not just some guy who throws prompts at a black box and hopes something good comes out  it was one of the most useful talks ive seen about LLMs  and i'm totally inspired to apply this to my work  seriously  if you're building anything with LLMs check out pydantic  you won't regret it
