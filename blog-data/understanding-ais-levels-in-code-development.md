---
title: "Understanding AI's Levels in Code Development"
date: "2024-11-16"
id: "understanding-ais-levels-in-code-development"
---

dude so i just saw this rad talk about levels of code ai it was like totally mind-blowing  the whole thing was about how they at source graph are thinking about ai in coding and it's way more nuanced than just "ai writes code" which is like super basic  they've got this whole framework they call "levels of code ai" and it's awesome

the basic gist is they're mapping ai assistance in coding to the sae levels of driving autonomy you know like self-driving cars level 0 is no ai level 5 is full-on robot overlords writing your code for you  it's a pretty clever analogy and it makes understanding this whole code-ai thing way easier  it's all about understanding the different stages of ai involvement in coding

first off they mentioned this stat that like 92% of devs use ai coding tools now which is bananas  a year ago it was just 1%  they even made a bold prediction that in 5 years 99% of code will be written by ai  which is kinda freaky but also kinda awesome maybe not awesome for devs but awesome for tech overall  that's some seriously next-level stuff

 so the levels themselves are split into three categories human-initiated ai-initiated and ai-led and each has two levels each for a total of six

level 0: this is the old school way total manual coding no ai help  it's like driving a car with no assistance its all you baby  the only thing close to ai is maybe some basic ide autocompletion

code example for level 0:

```python
# a simple function to add two numbers completely manual no ai assistance whatsoever
def add_numbers(x y):
    """this function adds two numbers together"""
    return x + y

# testing the function also completely manual
result = add_numbers(5 3)
print(f"the sum is: {result}") #prints 8
```


level 1:  this is where things start getting interesting you're still in charge but the ai is starting to help  think of it as like cruise control in a car it helps but you're still driving  the ai generates lines or blocks of code based on what you're doing  it's learning from tons of open-source code so it can suggest better completions

level 2:  now we're talking the ai understands the context of your codebase way better it's not just guessing anymore it knows what libraries you're using and can suggest completions based on that it's like having a super smart pair programmer who's seen all the code ever written  this is like a car with adaptive cruise control and lane keeping assist  it's doing more but you are still completely in control

code example for level 2 (using context):

```javascript
// imagine you're using axios in nodejs
const axios = require('axios');

//level 2 ai would know this and help autocomplete this request
axios.get('https://api.example.com/data')
  .then(response => {
    console.log(response.data); // ai would know data is a property of response in axios
  })
  .catch(error => {
    console.error(error); //ai would help with error handling in axios
  });
```

level 3:  holy moly we're getting into serious ai territory here  you give it high-level requirements like "add user authentication" and the ai builds the whole thing it generates the code explains how it works and integrates it  this is like a self-driving car that can handle most driving situations but needs human intervention if something unexpected happens it is capable of full components and integrations

level 4:  the ai is now proactively handling coding tasks without you even needing to ask  it monitors your code for bugs automatically submits pull requests to fix them and even updates your documentation  it's like a fully self-driving car that can handle almost everything on its own you just need to review its work this level introduces the concept of proactive ai handling code improvements documentation and bug fixes completely autonomously

code example for level 4 (proactive bug fixing):

```python
# imagine the ai detects a potential bug like a missing error handling for a file read operation 
# this is after the fact modification to existing code this would also need to take into consideration existing code to avoid collisions
try:
    with open("my_file.txt", "r") as f:
        contents = f.read()
        # previous code was missing this check
        if contents:
            #process contents
            print(contents)
except FileNotFoundError:
    print("file not found") #ai would add the missing exception handling to improve resilience
except Exception as e: #ai would also add a general exception handler to catch unexpected issues
    print(f"an error occurred {e}")
```


level 5:  this is the singularity moment for code  the ai handles the entire software development life cycle from start to finish  you just give it requirements and it designs builds deploys and maintains the whole thing  it's full ai autonomy  it's like a level 5 self-driving car where you don't even need a steering wheel it drives itself completely autonomously

so yeah that's the levels of code ai in a nutshell it's a really cool framework for thinking about how ai is changing software development and it's definitely something to keep an eye on  the talk even had some visuals of cars with different levels of autonomy side by side with  visual representations of code  and it really helped make the connection clear  it was all pretty insightful and i thought the analogies were really well done i mean who doesn't love a good car analogy right


the overall takeaway is that ai in coding is evolving rapidly and we're moving towards a future where ai will play a much larger role in the software development process  it's not about replacing developers but rather augmenting their capabilities and allowing them to focus on higher-level tasks  and honestly it’s pretty awesome to think about
