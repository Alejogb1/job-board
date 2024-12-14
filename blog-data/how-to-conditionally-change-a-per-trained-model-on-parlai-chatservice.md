---
title: "How to Conditionally change a per-trained model on ParlAI chat_service?"
date: "2024-12-14"
id: "how-to-conditionally-change-a-per-trained-model-on-parlai-chatservice"
---

ah, so you're looking into dynamically adjusting a pretrained model within parlai's chat service, huh? been there, done that, got the slightly singed t-shirt. it's a bit of a rabbit hole, and you definitely need to get your hands dirty with the parlai api. let me share some of my scars, err, experience.

basically, you’re not going to be changing the model's underlying architecture mid-conversation. that's not how these things usually roll. instead, what we're really talking about is modifying the model's behavior by altering its inputs or outputs based on certain conditions. think of it like you're giving it extra instructions based on the context of the ongoing chat.

i remember one time, way back when i was working on a chatbot for a customer service scenario, i needed it to handle angry customers differently. if someone was clearly frustrated (using very specific keywords that i made sure to capture with some text analysis), i needed the bot to switch to a more apologetic and less assertive tone. this wasn't about changing the model’s core weights, it was about changing the *context* i presented to the model, kind of pre and post-processing the interaction, you could say.

here's the thing with parlai: you've got the `world` object which handles all the interactions between the agents (your bot and the user). and then, you've got the agent itself, which is usually where the model lives. what you want to achieve is basically to intercept what the user says, check for the conditions you have defined, and then, pass a modified input to the agent and/or alter its output before presenting it to the user.

i’m thinking you’re looking at implementing this at the agent level or in a wrapper around it, which is something i found more manageable than trying to rewrite parts of the `world` itself.

let’s break down a typical approach. the core idea is to wrap your agent with a conditional logic layer. this layer inspects the incoming `observation` (the user input and the chat history) and the response the agent is about to generate. based on certain criteria, you can then modify both.

let's start with a basic example. let’s say you want to have your agent respond with "i am sorry" whenever the user says "bad".

```python
from parlai.core.agents import Agent
from parlai.core.message import Message

class ConditionalAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.wrapped_agent = create_agent_from_opt(opt) # you would make your actual agent here

    def observe(self, observation):
        if 'text' in observation and 'bad' in observation['text'].lower():
            observation['text'] = 'i am sorry'
        
        # pass the (potentially) modified observation to the actual agent
        return self.wrapped_agent.observe(observation)

    def act(self):
      response = self.wrapped_agent.act()

      # you could also modify the response here
      # lets say you want to append some text in a specific situation
      if 'text' in response and 'i am sorry' in response['text'].lower():
        response['text'] = response['text'] + ", how may i help?"

      return response

def create_agent_from_opt(opt):
  from parlai.agents.transformer.transformer import TransformerAgent
  agent = TransformerAgent(opt) # or whatever agent you are using
  return agent
```

in this example, the `observe` method checks the incoming text and replaces it if "bad" is found. then it allows the actual agent to respond. then in the `act` method, we further add text after the actual agent responds. it's a very simple conditional response, but it demonstrates the core idea. this is the most straight-forward thing you can do, although it's very rigid since it replaces the user text.

now let’s get a little more sophisticated. imagine you want to alter the behavior of the model based on the *sentiment* of the user input. for this, you might need to add a sentiment analysis component. let’s add it as a feature to the class.

```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from parlai.core.agents import Agent
from parlai.core.message import Message

class SentimentConditionalAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.wrapped_agent = create_agent_from_opt(opt) # your actual agent
        nltk.download('vader_lexicon', quiet=True)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def observe(self, observation):
        if 'text' in observation:
          sentiment_score = self.sentiment_analyzer.polarity_scores(observation['text'])['compound']
          if sentiment_score < -0.2 : # some arbitraty value
            observation['text'] = "oh my, i'm really sorry to see that you are sad, i will be more helpful"
        # pass the (potentially) modified observation to the actual agent
        return self.wrapped_agent.observe(observation)

    def act(self):
      response = self.wrapped_agent.act()
      # we will only modify if it was in negative sentimetn before.
      if 'text' in response and "oh my, i'm really sorry to see that you are sad, i will be more helpful" in response['text'].lower():
        response['text'] = "i will try harder."
      
      return response

def create_agent_from_opt(opt):
  from parlai.agents.transformer.transformer import TransformerAgent
  agent = TransformerAgent(opt) # or whatever agent you are using
  return agent
```

in this example, the `observe` method includes a very simple sentiment analysis using nltk's vader library, and based on the sentiment, if it falls below a score, it substitutes the text with a pre-defined sorry text, before calling the agent, then in the `act` method it checks whether the previous text was "oh my, i'm really sorry to see that you are sad, i will be more helpful" and if so substitutes it for "i will try harder.". this is obviously not great sentiment analysis but it shows that you can conditionally pre-process your inputs.

now, you might think this is a bit crude, and it is. ideally, you'd integrate your sentiment score directly as a feature that could influence the model's response and its output (using a wrapper to access both the input and output). you can do it by introducing a custom agent subclass or wrapper that does something of this sort.

but let's move to another example. let's imagine you want to use a different type of processing based on a previous turn. in this case, we can add this conditional logic into the `act` method.

```python
from parlai.core.agents import Agent
from parlai.core.message import Message

class HistoryConditionalAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.wrapped_agent = create_agent_from_opt(opt) # your actual agent
        self.last_response = None

    def observe(self, observation):
        # pass the observation to the actual agent
        return self.wrapped_agent.observe(observation)

    def act(self):
      response = self.wrapped_agent.act()
      if self.last_response and 'text' in self.last_response and 'question' in self.last_response['text'].lower():
        if 'text' in response:
           response['text'] =  "well, i understand that " + response['text'] + "but you asked a question" 
      self.last_response = response
      return response

def create_agent_from_opt(opt):
  from parlai.agents.transformer.transformer import TransformerAgent
  agent = TransformerAgent(opt) # or whatever agent you are using
  return agent

```
here, the `HistoryConditionalAgent` keeps track of the previous response. and if the previous response included the word "question" it will add "well, i understand that X but you asked a question" to the next answer given by the agent. so you are conditioning the output based on the previous agent output.

the interesting thing here is that you can basically do arbitrary things and condition the whole conversation from arbitrary rules that you might want to set.

now, a quick bit of advice, when you’re doing this kind of conditional logic, you need to be careful to not get your logic too convoluted and hard to debug. i once tried to make a bot that had like 10 different conditions. it was a debugging nightmare. it started changing the answer based on the average number of vowels the person was saying… a joke, i know i know, but it felt like it. simpler is often better.

for further reading, i would recommend checking out papers on contextual bandit algorithms. even though we are not strictly talking about reinforcement learning here, they have great methods to condition on the state of the environment. also look for research on 'policy-based dialogue systems'. that should give you an idea of the kind of things you could do with the response generated by your model.

oh, and one more thing, always test your conditions really thoroughly. it is very easy to introduce subtle bugs that are only triggered on specific edge cases. and there's nothing worse than a chatbot that starts acting like it's having a stroke because of one tiny overlooked detail.

anyway, hope this helps. let me know if you have any more questions. i'm usually lurking around these parts.
