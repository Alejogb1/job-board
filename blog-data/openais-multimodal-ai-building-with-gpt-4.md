---
title: "OpenAI's Multimodal AI: Building with GPT-4"
date: "2024-11-16"
id: "openais-multimodal-ai-building-with-gpt-4"
---

dude so i just watched this totally rad talk by roman from openai and lemme tell you it was a wild ride like seriously  the whole thing was about how openai is super stoked about all these new ai models and how they're letting devs build some next-level stuff  it's like they're saying "hey we made some awesome toys come play with them"  but way more techy and with way less glitter

first off the whole vibe was super chill like roman was just casually hanging out on stage  it wasn't your stuffy corporate presentation type deal  he straight up started by asking the crowd "who's played with the gpt3 api" and half the room went nuts  that was a pretty cool visual cue right there showing how many people are already on board with openai's stuff  another visual cue that totally stuck with me was when he drew a really bad picture of the golden gate bridge  and gpt-4  just *knew* where he was  it was like magic lol  then there was this audio thing where he used his own voice to create a custom voice using openai's voice engine tech— that was mind blowing

so the main point was openai isn't just about chatgpt anymore they're all about multimodal models  that means their models can deal with text images audio video  the whole shebang  like they're throwing the kitchen sink at this thing  and the key idea here is that combining these different types of data lets you make ai experiences that feel way more natural and intuitive that's a big step away from just typing commands into a chat box

he talked a lot about gpt-4 its improvements and how much faster and cheaper it is  they even mentioned gpt-4 is twice as fast as gpt-4 turbo and half the price  that's insane efficiency gains for developers  it's like getting a supercharged computer for free  and they're working to get rid of rate limits entirely which is huge news for anyone building things at scale  this was a big deal because it means developers can build bigger cooler things without constantly worrying about hitting limits

another huge concept was the focus on multimodal interactions  they showed off a live demo using gpt-4  on a desktop app  it was incredible seeing it seamlessly handle text audio and visual inputs in real time  it was like talking to a really smart friend who also has x-ray vision  and super hearing  they can answer questions look at images even interpret drawings  and this tech is being used in stuff like generating playlists in spotify  that’s how it's being used in the real world

then he dropped this killer code snippet showing how gpt-4 helped fix the responsiveness issues in a react app  i'm talking clean code beautiful ui no more squished images on mobile it's what every dev dreams of

```javascript
// before -  not responsive
const Discover = () => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {/* your travel spots */}
    </div>
  );
};

//after gpt-4's help
const Discover = () => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {/* travel spots */}
    </div>
  );
};
```

pretty straightforward right  but that's the power of having an ai assistant that understands your code  he just pasted some code and gpt-4 instantly identified the issue and provided the fix using tailwindcss  it used media queries which is a basic concept  but still amazing how it just pinpointed it  this is beyond simple code completion  this is true ai-assisted development

another snippet showcased using react hooks with gpt-4 for a chat application  this one was focused on backend integration but still useful

```javascript
import React, { useState, useEffect } from 'react';

const useAssistant = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (message) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/assistant', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      setMessages([...messages, message, data.response]);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };
  // ... rest of the hook for file uploads and streaming responses
  return { messages, sendMessage, isLoading };
};

export default useAssistant;
```

this one is more complex  it involves fetching data from a server handling loading states  and managing the conversation history  but again showing how gpt-4 helps simplify even more complex code

and one more  this time it’s about using the openai api directly

```python
import openai

openai.api_key = "YOUR_API_KEY"

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003", #or any other suitable model
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    text = response.choices[0].text.strip()
    return text

user_prompt = "Write a short story about a robot learning to love"
generated_story = generate_text(user_prompt)
print(generated_story)
```

simple right  but this is the foundation of so many ai applications  this snippet demonstrates how you use the openai api to generate text  you just give it a prompt and it spits out some text  it’s the groundwork for everything from chatbots to creative writing tools  and all you need is an api key to get started

the talk ended with roman talking about their future plans  they're focusing on making models even smarter faster and cheaper  customizing models for specific use cases and building more sophisticated agents that can interact with the world using all these modalities  it's like they're building a whole new universe of ai possibilities  a universe where ai is so easy to use that even your grandma could build a killer app  well maybe not your grandma but you get the picture

basically the whole talk was a giant hype train for the future of ai  and honestly i'm on board  roman ended by saying "it's the most exciting time to be building ai-native companies" and i couldn't agree more  it's all about building stuff without limits  it’s about using ai to make things that were previously impossible  and that's pretty awesome
