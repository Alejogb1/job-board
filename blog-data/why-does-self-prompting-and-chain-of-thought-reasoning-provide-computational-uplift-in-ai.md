---
title: "Why does self-prompting and chain of thought reasoning provide computational uplift in AI?"
date: "2024-12-11"
id: "why-does-self-prompting-and-chain-of-thought-reasoning-provide-computational-uplift-in-ai"
---

Okay so you wanna know why self-prompting and chain of thought reasoning are like supercharging AI right  its kinda cool actually  imagine you're trying to solve a really hard math problem  you wouldn't just jump to the answer right you'd break it down into smaller easier steps  that's basically what chain of thought is all about  it's like giving the AI a little internal monologue to guide its thinking process  instead of just spitting out the first thing that pops into its head it meticulously walks through each step  reasoning its way to the solution

Self-prompting is even more meta its like the AI is giving itself hints or questions along the way its  basically  a form of self-teaching  it prompts itself to consider different aspects of the problem  explore alternative approaches  and generally be more thorough and less prone to making stupid mistakes  think of it as a built-in debugger or a super smart tutor that's always whispering helpful suggestions in the AI's ear

The computational uplift comes from this more structured and deliberate approach  it's not just about brute force computation  its about smarter computation  by breaking down complex problems into smaller more manageable chunks the AI can allocate its computational resources more efficiently  it's like optimizing your code for better performance  instead of using a giant messy function you refactor it into smaller more modular functions that are easier to understand debug and optimize

Before chain of thought models  large language models often struggled with complex reasoning tasks  they might get simple things right but complex multi-step problems were a different story  chain of thought helps bridge that gap  it allows the model to handle problems that were previously out of its reach  its like unlocking a whole new level of capability

Think of it like this  you have a huge maze to solve  without chain of thought the AI might randomly wander around bumping into walls  with chain of thought  the AI systematically explores different paths  eliminates dead ends and eventually finds its way to the exit  it's a more efficient and effective strategy

Now  some people might think  well  isn't this just making the AI slower  because it's doing more steps  and  yeah  sometimes it can be slightly slower but the increase in accuracy and the ability to solve more complex problems often far outweighs the small performance hit its a trade off between speed and accuracy and in many cases the accuracy win is much more valuable

The computational uplift isn't just about speed  it's also about capability  its about expanding the range of problems the AI can successfully solve  its like upgrading your computer  you might not get a massive speed boost but you can now run more demanding applications  you've increased your overall computational power even if its not reflected purely in clock speed

Lets look at some code examples to illustrate this a bit better  though remember these are simplified illustrations the actual implementation in state-of-the-art models is much more complex

**Example 1: Simple Arithmetic**

Without chain of thought a simple model might struggle with something like "What is 3 + 7 * 2 - 5?"  it might just give a random answer or make a mistake with order of operations but with chain of thought it can break it down:

```python
problem = "What is 3 + 7 * 2 - 5?"
# Chain of thought reasoning
steps = [
    "First we do multiplication: 7 * 2 = 14",
    "Next we do addition: 3 + 14 = 17",
    "Finally we do subtraction: 17 - 5 = 12",
    "Therefore the answer is 12"
]
answer = 12
```


**Example 2: Common Sense Reasoning**

A simple question like "The bird is in the cage the cage is in the house where is the bird?" might be difficult without chain of thought  With chain of thought:

```python
problem = "The bird is in the cage the cage is in the house where is the bird?"
# Chain of thought reasoning
steps = [
    "The bird is in the cage",
    "The cage is in the house",
    "Therefore the bird is in the house"
]
answer = "The bird is in the house"
```


**Example 3: Symbolic Reasoning**

Let’s say we have a more abstract problem: "All men are mortal Socrates is a man Is Socrates mortal?"

```python
problem = "All men are mortal Socrates is a man Is Socrates mortal?"
# Chain of thought reasoning
steps = [
    "Premise 1: All men are mortal",
    "Premise 2: Socrates is a man",
    "Conclusion: Therefore, Socrates is mortal (deductive reasoning)"
]
answer = "Yes"

```

These are basic examples but they highlight the core idea  chain of thought allows the model to decompose problems  break down complex tasks into a series of simpler steps and ultimately solve problems it couldn't have solved before  its like giving the AI a toolbox full of reasoning techniques and showing it how to use them effectively

For further reading check out papers on neural symbolic reasoning and  look into some books on cognitive psychology and artificial intelligence that dive into the mechanisms of human reasoning  those will give you a much deeper understanding  and you'll find lots of relevant papers if you search for "chain of thought prompting" "self-prompting" and "large language models" on academic search engines  good luck  its a fascinating area  and there's so much more to explore  have fun  I hope this helped a bit  let me know if you wanna dive deeper into any specific aspect
