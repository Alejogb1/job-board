---
title: "GitHub Copilot: AI-Assisted Coding"
date: "2024-11-16"
id: "github-copilot-ai-assisted-coding"
---

dude so i just watched this amazing talk by thomas domke the github ceo seriously mind-blowing stuff he was chatting about github copilot and the future of ai in coding and it was like a rollercoaster of insights  it’s all about how ai is changing how we write code and collaborate  think less about typing and more about *thinking*  the whole thing was like a wild ride and i gotta share it with you

first off the setup was killer  they introduced thomas as this coding legend who’s been obsessed with software since he was a kid in germany  then BAM the whole thing shifts to copilot this ai-powered coding assistant that's basically changing the game  i mean  the guy’s building tools that are literally reshaping how software gets made that’s next level

one of the key moments was when thomas talked about the very first time he saw copilot in action  it was 2020 lockdown remember everyone was stuck on zoom calls  he described this moment where they were just playing around with this early version of gpt-3 codex from openai and  this is the money quote “we were dictating prompts and asking the model to write some code and i think the first aha moment that i had is that you could ask it to write javascript code and put the curly braces in the right places”  just imagine being there  witnessing the birth of something revolutionary  the visual cue was him on a zoom call totally focused  and i could practically *feel* that giddy excitement

another big moment was the internal testing  the net promoter score that’s nps for the nerds  was off the charts like 72 or 73  totally unheard of for a new product especially one using this cutting edge language model technology and ui stuff is a whole other beast  it was this wild validation that this ai thing wasn't just a gimmick

a third key moment was the telemetry data  at first they thought it was buggy  it was showing that copilot was writing like 25% of the code it was just too crazy to believe  but then they validated it and it was true the thing was actually writing a quarter of all the code  and it just kept climbing  now it’s closer to half in some cases even more with languages like java this is massive implications

a fourth amazing thing he talked about was how github is trying to make copilot super integrated into their whole platform  not just a simple autocompletion tool  it’s about  “meeting the developer where they are”  he mentioned that copilot is evolving beyond simple autocompletion to assisting with tasks from concept to completion within github  it's going way beyond the ide

finally the big picture vision was awesome  thomas talked about moving beyond the ide and towards natural language coding you know  talking to the computer in plain english to create code  this is super cool  he even mentioned that copilot handles multiple languages making coding more accessible to people worldwide and that’s what i call a good mission

now let's dive into some technical details  remember that  “aha” moment he talked about  here's a little python snippet that showcases that same kind of magic

```python
# simple example of copilot-like functionality
def greet(name):
    """greets the person passed in as a parameter"""
    print(f"hello {name}!")

# copilot would suggest the following based on context
greet("bob")  # copilot would probably suggest this line  or even the whole function
```

this is a really basic function  but imagine writing a complex algorithm  and copilot is suggesting code snippets  or even entire functions  that’s what i’m calling next level  it’s like having a super smart pair programmer sitting next to you

he also talked about the challenge of navigating large codebases  that's where github copilot workspace comes in  it's bridging the gap between tasks and code  i mean think about this

```python
# Imagine a github issue: "Fix bug in user authentication"
# Copilot Workspace analyzes the codebase and suggests:
# 1. Identify the authentication function: auth.py:check_password()
# 2. Investigate the database query: db.py:get_user_data()
# 3. Add logging to track user input: logging.py:log_authentication()
# 4. Implement additional input sanitization: auth.py:sanitize_input()
# 5. Write unit tests for the fix: tests/auth_test.py
# The workspace shows how this maps to changes and diff views within the project
```

it’s showing you the steps to fix a bug  identifying the relevant files  even providing a plan with the changes as diffs  it's not just suggesting code  it's guiding you through the entire process  this is huge for anyone working on a big project or a new code base

and of course we need some javascript  it's just too fun not to  here’s a little snippet illustrating how copilot could make front-end development a lot less tedious

```javascript
// imagine you're building a simple to-do list
const todoItems = [];

// copilot might suggest this for adding a new item
function addItem(item){
    todoItems.push(item);
    renderList();  // copilot intelligently infers you need a rendering function
}

// copilot would likely generate this function too
function renderList() {
    const list = document.getElementById("todo-list");
    list.innerHTML = ''; // clear existing list
    todoItems.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
    });
}
```

see how it even suggests the `renderList()` function  based on the context of adding an item  that’s pretty insane  it's basically automating a lot of the boilerplate and repetitive code  that’s just plain awesome

the resolution of the whole talk  in my opinion  is that ai is not here to replace developers  but to empower them  to make them more productive  to free them up to focus on the creative and challenging parts of software development  it's about augmenting human capabilities  not replacing them   the tools are becoming more intuitive  less about complex commands  and more about natural language  this shifts the dynamic for better collaboration between developers and non-developers  it's basically changing the way we think about building software  and it's only getting started  which is both exciting and totally bonkers
