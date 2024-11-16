---
title: "GitHub Copilot:  AI-Assisted Coding's Future"
date: "2024-11-16"
id: "github-copilot--ai-assisted-codings-future"
---

yo dude so i just watched this killer talk by mario rodriguez the vp of product at github about copilot and the future of ai in coding and man it was awesome  super insightful and hilarious too  it was basically a behind-the-scenes look at how they built this thing and where it's headed next and let me tell you it's wild

the whole point of the talk was to give us a peek into the making of github copilot this revolutionary ai coding assistant  mario spent a good chunk of time talking about the challenges and triumphs of creating a product like this at scale its impact and the future of ai-assisted coding  think of it as a deep dive into the sausage factory of ai development only way more fun and less greasy

first off mario dropped this insane stat copilot is used by over 20000 orgs and a million plus devs  like whoa that's a lot of people using something he and his team built  he mentioned it in a really casual way too like "oh yeah and a million devs use this thing no biggie" which made me laugh because you could tell that it was a huge accomplishment

then he got into the juicy details like how the whole thing started with a 2020 paper called "an automated ai programmer fact or fiction"   that's where the whole "polarity" concept came up  he described it as the idea that the choice of fact or fiction wasn't mutually exclusive  they were  intertwined like he mentioned having kids who think it's a fact that dinosaurs exist, regardless of the scientific reality  funny and thought provoking

one of the key ideas he highlighted was the absolute importance of ux  user experience  he stressed that the way you present an ai tool can totally make or break it  he said that the initial attempts to integrate copilot into the pull request workflow were "not good" it's like finding out your favorite band decided to use a kazoo as their only instrument  it's not the same  they made it work eventually though

he talked about four crucial components to copilot v1's success:

1 ghost text holy moly this is a big one  he explained how having that little "suggestion" preview completely changed the game  it allowed users to see what the ai was suggesting before it committed the code changes like a preview screen for the next suggestion from the AI  it completely changes the user experience making it super intuitive

2 speed  he emphasized that speed is key if you're trying to keep devs in the flow  nobody wants to wait seconds for an AI to suggest a line of code  like waiting at the store for your favorite snack, it sucks the fun out of everything so copilot needed blazing-fast speed

3 the model itself he gave props to openai for creating the codex model (they use gpt 3.5 turbo now) saying it was a game-changer  it was way better than anything they'd seen before it was a perfect fit in this context and what they needed to move forward

4  prompt engineering  he even mentioned a job opening for people skilled in prompt engineering because it takes a ton of expertise to craft prompts that consistently elicit quality suggestions from the ai  just creating a prompt that generates millions of lines of code takes skills


here's a little python snippet illustrating the idea of ghost text  imagine a simple code completion scenario

```python
# imagine the user is typing this
def greet(name):
    # ghost text appears here before the user finishes typing
    print(f"Hello, {name}!") # the AI predicted the rest


# and if the user doesn't like it
def greet(name):
    # user types something else
    print("Hi there," + name + "!")

```


mario then talked about scaling copilot  this is where things got even more interesting  he highlighted a few challenges they faced:


1  syntax isn't software  just because an ai can understand the syntax of a language doesn't mean it automatically gets how to write good software   it's like understanding grammar vs. writing a novel  it's about the logic and structure too and the semantics of the overall task

2 global presence to maintain that super-fast speed for all users copilot needs servers all over the world  he mentioned deployments in japan europe and multiple north american data centers  that's a lot of infrastructure

3 offline vs online evaluations  what works in testing doesn't always work in real-world conditions  they needed to establish comprehensive scoring mechanisms to track performance in both environments  and then use this to iterate quickly

here's some pseudocode illustrating how you might represent offline/online evaluations:

```python
#simplified example, the actual scorecard implementation is complex
class Scorecard:
    def __init__(self):
        self.offline_metrics = {}
        self.online_metrics = {}

    def add_offline_metric(self, metric_name, value):
        self.offline_metrics[metric_name] = value

    def add_online_metric(self, metric_name, value):
        self.online_metrics[metric_name] = value


#example usage
scorecard = Scorecard()
scorecard.add_offline_metric("accuracy", 0.98) #Example accuracy from offline testing
scorecard.add_online_metric("latency", 120) #latency in milliseconds from online testing
scorecard.add_offline_metric("user_satisfaction", 4.5) #on a 5-star scale

print(scorecard.offline_metrics)
print(scorecard.online_metrics)

#This would be incorporated into the testing pipeline and used to track performance
```

another awesome point was the importance of responsible ai he stressed things like security (don't store user data at rest) legal compliance and ethical considerations this isn't just about building cool tech it's about building it responsibly which is way more important than anything else

and finally mario looked at the future  he talked about shifting from procedural coding (step-by-step instructions) to goal-oriented programming (defining goals and constraints)  this is where things got really futuristic he mentioned a brendan victor talk (a must watch btw) about this very idea

here's a javascript snippet giving you a taste of this  a goal-oriented approach using a hypothetical api call


```javascript
// Instead of specifying every step manually, we describe the desired outcome
const goal = {
  type: "create_webpage",
  content: {
    title: "My Awesome Website",
    sections: [
      { type: "introduction", content: "..." },
      { type: "about", content: "..." }
    ],
  },
};

// Hypothetical function that uses AI to achieve the goal
const webpageCreator = async (goal) => {
  const generatedHtml = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(goal)
  }).then(res => res.text());

  //Further actions based on the generatedHtml such as saving it to a file or loading it onto a server
  return generatedHtml;
};


webpageCreator(goal).then(html => console.log("Html generated:", html));

//The webpage is created automatically based on our description of the goal
```

he also discussed the need for ai to reason about code using techniques like summarization and pattern recognition like your brain effortlessly does and creating a coding environment optimized for human-ai collaboration  no more sidebars buddy  imagine a truly integrated experience that changes how people and AI work together  the immersive environment that he talked about seemed cool and interesting

the whole talk was a whirlwind of insights challenges solutions and future aspirations  it wasn't just about copilot its about the future of coding itself and the way we interact with technology  mario’s talk was the perfect blend of tech talk and stand up comedy making it a seriously entertaining and educational experience
