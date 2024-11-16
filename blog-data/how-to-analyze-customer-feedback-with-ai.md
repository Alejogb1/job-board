---
title: "How to Analyze Customer Feedback with AI"
date: "2024-11-16"
id: "how-to-analyze-customer-feedback-with-ai"
---

dude so this video was like a total rollercoaster of ai awesomeness and also some serious customer service drama i mean who hasn't had a terrible customer service experience right  the whole point was showing off echo ai this new platform that uses generative ai to basically become a superpowered customer service analyst think of it like this imagine you're drowning in a sea of customer chats emails and phone calls  you're trying to find the needles in a haystack but there's just too much stuff  echo ai is like a giant net that scoops up everything and then uses ai magic to sort through it all  finding hidden problems insights and trends you would never notice otherwise seriously it's way beyond just keyword searching

they started by painting a super relatable picture of the typical enterprise struggle they said something like "just the tip of the iceberg" that's the phrase that stuck with me remember that visual  they were talking about how companies get huge and lose track of what their customers are actually saying and feeling it's not that they don't care it's just impossible to keep up with the sheer volume of interactions

another key moment was this great analogy about how companies traditionally deal with customer feedback they showed this three-column chart illustrating the progression from manual reviews which are super inaccurate and time-consuming then to building complex scripts for retroactive analysis and finally to the current state-of-the-art which is what they're trying to solve with echo ai it's like they're saying "we're moving from putting out fires to preventing them entirely"

the whole thing is built on lms large language models but they didn't just throw an lms at the problem they built a whole pipeline and the key concepts here were this "100% coverage" thing and the importance of trust the 100% coverage part is pretty straightforward instead of sampling only a tiny fraction of customer interactions echo ai analyzes everything this lets them find unexpected problems trends and patterns  but the trust bit is really interesting because companies are super worried about relying on ai for important decisions  so they went out of their way to show how they’re constantly working to build trust with their clients through accuracy and transparency and this is where log10 comes in

log10 is their secret sauce for accuracy because seriously accuracy is like the holy grail of ai especially in this field ai can sometimes hallucinate things completely make stuff up  like the air canada chatbot incident they mentioned the one that promised a fake refund thats a nightmare  or the chevy tahoe chatbot selling a truck for a dollar that's absurd  log10 helps them measure and improve the accuracy of their ai models it's like a quality control system on steroids they explained they've been using it to monitor and triage model problems and get feedback to constantly improve

they showed this really cool graph it was comparing the predicted feedback vs the actual human feedback when using log10 and when not  it was like night and day the one without log10 had this random scatter plot showing the ai's predictions were totally all over the place the log10 one had a strong positive correlation super accurate predictions this was the big sell for their system it’s not just about having ai it’s about having accurate ai

so here’s where i get to show off my coding skills because this is where it got really fun they actually included code snippets  ok maybe not exactly snippets but i can totally write something similar based on what they talked about


first a simple python snippet showing how you might use an lms to analyze customer sentiment:


```python
import openai

openai.api_key = "YOUR_API_KEY"

def analyze_sentiment(customer_message):
  response = openai.Completion.create(
    engine="text-davinci-003", # or another suitable model
    prompt=f"Analyze the sentiment of the following customer message:\n{customer_message}\nIs the sentiment positive, negative, or neutral?",
    max_tokens=50,
    n=1,
    stop=None,
    temperature=0.5,
  )
  sentiment = response.choices[0].text.strip().lower()
  return sentiment

customer_message = "i'm so frustrated your website is down again"
sentiment = analyze_sentiment(customer_message)
print(f"customer sentiment: {sentiment}") # outputs something like: customer sentiment: negative


```

this is a super basic example but it illustrates the core concept of using an lms to extract information from text  you could easily expand this to include other analyses like topic extraction intent recognition or even generating summaries


then they also talked about how they connect to various systems to gather data  here's a very basic python snippet simulating that process

```python
# this is a highly simplified representation of data ingestion

import json

def get_data_from_system(system_name, api_key):
    #Replace with your actual API call and error handling
    if system_name == "zendesk":
        # simulating an API call
        data = {"tickets": [{"id": 123, "message": "my wifi is down"}, {"id": 456, "message": "Your product is amazing!"}]}
        return data
    elif system_name == "salesforce":
        #Simulating a different api call
        data = {"chats": [{"id": 789, "message": "i need help with my order"}, {"id": 101, "message": "i love your service! thank you!"}]}
        return data
    else:
        return {"error": "system not supported"}

zendesk_data = get_data_from_system("zendesk", "zendeskApiKey")
salesforce_data = get_data_from_system("salesforce", "salesforceApiKey")

print(json.dumps(zendesk_data, indent=2))
print(json.dumps(salesforce_data, indent=2))

# in a real application you will need robust error handling, proper auth, and sophisticated data structures
```


finally this whole thing culminates in a demo showing their platform in action  they showed a bunch of generated insights from customer conversations  things like sentiment analysis topic extraction and even automatically generated summaries of calls and this is where the magic of log10 really shines through

log10 is a feedback loop it's constantly evaluating the models accuracy and using that feedback to fine-tune them they used the example of summarization showing how the system scores the summaries and allows human override this constant feedback loop is what separates echo ai from other ai-powered solutions  it is constantly learning and improving making it even more accurate over time

so the resolution is pretty clear  echo ai offers a powerful way for companies to unlock the hidden insights buried in their customer interactions by providing a high-accuracy system that is constantly learning and improving  it solves the problem of dealing with massive amounts of data and allows businesses to make data-driven decisions about their product or service improvements this is not just about generating insights its about using generative ai to create trust in the process

i know it’s a lot but man this is what’s happening in the ai world right now it's a combination of raw lms power clever engineering and a constant drive for accuracy and trust it’s super exciting stuff and it’s only gonna get crazier from here  plus i got to sprinkle in some python code so win-win  what do you think
