---
title: "LLM Development Best Practices: A  Future Courtroom Tutorial"
date: "2024-11-16"
id: "llm-development-best-practices-a--future-courtroom-tutorial"
---

dude so this video right it’s like this totally wild courtroom thing  the whole shtick is this llm judge maximus from 2034 –  a future where ai judges are a thing apparently – is presiding over this ai engineer judgment day.  it’s all a bit of a skit but it’s also packed with legit advice for folks building stuff with llms. like  the whole point is to highlight common mistakes and best practices. think of it as a hilarious, futuristic  pep talk with some seriously useful tech tips thrown in.  we’re talking about real-world  llm development, not just some abstract theory.


first off the visuals are killer maximus the llm judge is this totally deadpan digital avatar like straight out of a cyberpunk movie.  alex the human commentator though he’s this super energetic guy, totally the opposite vibe he’s basically the  "human in the loop"  commenting and adding some real-world context to the absurd legal proceedings. and there’s this awesome running gag about the judge getting interrupted. classic. the whole thing is presented with this kind of mock seriousness which makes the whole thing funnier.


so like five key moments i’d say the first is the "no trace left behind"  verdict.  this dude daniel builds a cool llm rapper at a hackathon deploys it to production without logging or tracing anything  then things break and he’s clueless.  major fail.   that’s why logging and tracing are essential even in the simplest projects.  you need to understand what's going on under the hood before things go sideways.  think of it like this:


```python
import logging
import functools

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def trace_function_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__} with args: {args} and kwargs: {kwargs}")
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} returned: {result}")
        return result
    return wrapper


@trace_function_calls
def my_awesome_llm_function(prompt):
    # your complex llm logic here
    # ... lots of cool stuff ...
    return "this is the response to " + prompt


my_awesome_llm_function("What's the meaning of life?")

```

this super simple python decorator uses the `logging` module to show you every single function call, its arguments and the returned value.  it’s simple to implement but super powerful for debugging  –  especially when you're dealing with complex llm chains and multiple function calls. if you aren’t already logging your function calls you should be.


then there's the "premature fine-tuning"  crime sasha just dives straight into fine-tuning llama 3 on company data without even trying basic prompt engineering. that’s a big no-no  you gotta iterate on your prompts and figure out what works *before* you go messing with the model weights. think of it like building a house—you wouldn’t start painting before you’ve laid the foundation, right?


```python
# example showing simple prompt iteration

prompts = [
    "translate english to french: hello",
    "translate english to french: hello, how are you?",
    "write a short story about a talking cat",
    "write a haiku about nature",
    "summarize this text: [insert large text here]",
]

#iterative testing of different prompt approaches
for prompt in prompts:
    response = llm_model(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}\n")


#adjusting the prompts based on outputs


prompts2 = [
    "translate english to french using a formal style: hello, how are you?",
    "write a short story about a talking cat named Mittens who lives in Paris",
    "summarize this text focusing on the key arguments: [insert text here]",
]

for prompt in prompts2:
    response = llm_model(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}\n")

```

this code shows simple iteration on prompts. you see you initially  try a few different styles of prompts and then use this initial knowledge to modify the prompts – to receive more specific, refined responses.  you gotta experiment before fine-tuning. fine tuning is an expensive and complex process and it’s wasted on unrefined prompts.


another big one is the importance of llm evaluations.  francisco's chatbot disaster – rushed to production with only programmatic evaluations – is a cautionary tale.  you need a multi-faceted approach  programmatic evals are great for simple checks but you need human-in-the-loop evaluations and overall better criteria for complex interactions. you absolutely need to test your llm in the real world, not just in a perfect lab setting.


```python
# simple programmatic evaluation for numerical output
def evaluate_math_model(model, test_cases):
    results = []
    for case in test_cases:
        input, expected_output = case
        actual_output = model(input)
        results.append((input, expected_output, actual_output, actual_output == expected_output ))
    return results

test_cases = [("2 + 2", 4), ("10 - 5", 5), ("5 * 3", 15)]

# mock llm response (replace with your actual model)
def llm_model(prompt):
    # replace with actual llm call 
    # ... complex stuff to get result ...
    if prompt == "2 + 2": return 4
    if prompt == "10 - 5": return 5
    if prompt == "5 * 3": return 15

results = evaluate_math_model(llm_model, test_cases)
print(results)
```

this shows a basic programmatic evaluation that is fine for things like simple math problems but you also need to do human evaluations of more complex prompts and outputs.   this tests for very simple outputs. it's pretty dumb but shows the idea.


then there's the "out-of-the-loop"  conviction. morgan ignores all the new ai news and sticks with an older model.  bad move.  the ai landscape moves *fast* you gotta stay current. think of it as staying up to date on the latest programming libraries, operating systems, and trends. you need to understand the current trends to build effectively.


finally, the whole "validate your validators"  point is super important. the llm judge maximus isn’t perfect it has biases and needs human oversight.  even your evaluation systems can be flawed – you need to check for those biases and continually improve your methods. it’s iterative and crucial.


in short this video is brilliant. it’s entertaining, informative, and packed with practical advice that everyone building llm applications should pay attention to  it’s not just about coding it's about the whole process, from prompt engineering to evaluation and even staying up to date with the latest news.  it’s a wild ride but seriously worth watching.
