---
title: "Reinforcement Learning: Q-Learning Tutorial"
date: "2024-11-16"
id: "reinforcement-learning-q-learning-tutorial"
---

dude so i watched this totally rad video on, like,  building these amazing little programs that basically teach your computer to, get this, *think*  it's all about machine learning and this thing called reinforcement learning which is basically like training a puppy but way cooler because it involves algorithms and stuff

the whole point of the video was to show how you can make a computer learn to do things without explicitly telling it how it's kinda like showing a kid how to ride a bike—you don't give them a step-by-step manual you just kinda guide them and let them figure it out by trial and error except instead of a bike it's solving a maze or playing a game  it was pretty mind-blowing tbh


 so let's dive in  the video started with this super cute animation of a little robot navigating a maze it was seriously adorable  that was like the setup—showing us the problem we're trying to solve getting a computer to solve a maze without giving it the solution directly it was visually appealing and set the stage perfectly i mean the robot looked like it was wearing tiny little roller skates  it was freakin' adorable


then they jumped into explaining the whole reinforcement learning thing remember how i said it was like training a puppy well that's basically it you use rewards and punishments to guide the "agent"  (that's our little robot) towards the desired behavior  if it takes a wrong turn it gets a negative reward (think a little digital frown face)  if it takes a correct turn it gets a positive reward (think celebratory fireworks or something)  it's all about shaping its behavior through these little digital carrots and sticks


one key idea they hammered home was the concept of a "q-table"  this is basically a giant spreadsheet that keeps track of how good each action is in each state  so  let's say our little robot is at position (2,3) in the maze  the q-table would have an entry for that position  and then for each possible action (go up, down, left, right) it would store a number representing how good that action is based on past experience higher numbers mean better actions it's like a giant lookup table  it’s like this:


```python
q_table = {} # initialize an empty dictionary to store Q-values
#sample entry for a specific state and action
q_table[(2,3, 'up')] = 0.5 #this represents that moving 'up' from (2,3) yielded a Q-value of 0.5
q_table[(2,3, 'down')] = -0.2 # moving 'down' yielded a negative Q-value in that case
q_table[(2,3, 'left')] = 0.8  #moving 'left' yielded a higher positive value 
q_table[(2,3, 'right')] = -0.1 #moving 'right' yielded another negative value
```

it was pretty cool to see how this table gets updated over time as the robot explores the maze it learns which moves are more effective at getting it to the end  the q-table is constantly being refined through this learning process


another key concept was the epsilon-greedy strategy  this is how we balance exploration and exploitation  exploration is trying out new things even if they might not be the best known options exploitation is always doing what you already know works best  the epsilon-greedy strategy essentially says  "most of the time do what works best but sometimes try something new" epsilon is a number between 0 and 1  a higher epsilon means more exploration  a lower epsilon means more exploitation  it's like saying "80% of the time let's do what worked before but 20% of the time let's try something new maybe it's better" it's a smart compromise


the code they showed for this was something like this


```python
import random

def choose_action(q_table, state, epsilon):
    if random.uniform(0,1) < epsilon: # explore with probability epsilon
        return random.choice(actions) # choose random action from possible actions
    else: # exploit otherwise
        return max(q_table[state], key=q_table[state].get)  # choose action with highest Q value

```


this  snippet is a function which decides whether to explore or exploit in a given state  depending on the epsilon value  it's a simple but powerful idea


another really neat visual cue was the way they animated the q-table updating  it was like watching a brain slowly forming connections  you could actually see the q-values changing as the robot learned  it was mesmerizing honestly  it really brought home the point that this is a dynamic process the q table isn't a static thing—it’s constantly evolving



the third code snippet they showed was a bit more involved it dealt with updating the q-table itself using a technique called q-learning  basically  you update the q-value of the action that was taken using this formula:


```python
# q-learning update rule
q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * max(q_table[next_state]) - q_table[state, action])

#where:
#alpha is the learning rate (0 < alpha <=1) How much we adjust Q values based on new info
#gamma is the discount factor (0 <= gamma <= 1) How much we value future rewards
#reward is the immediate reward received for taking the action
#max(q_table[next_state]) is the maximum Q value in the next state

```

it's a bit dense but basically  it adjusts the q-value based on how much better the action was than expected  it takes into account the immediate reward and the potential future rewards it's all about adjusting values for future learning  it's like learning from mistakes and refining your actions


anyway  the resolution of the video was pretty straightforward  by using reinforcement learning  specifically q-learning  the robot successfully learned to navigate the maze without any explicit instructions  it showed how a simple algorithm can allow a computer to learn complex tasks  it wasn't just solving a maze it was demonstrating a powerful concept—that computers can learn to learn which is seriously cool


that's the gist of it dude  i really enjoyed the video  it was both educational and entertaining  it made a complex topic like reinforcement learning accessible and engaging  and yeah the little robot was just adorable  seriously check it out if you ever get a chance you know you wanna see that cute lil' robot rock a maze
