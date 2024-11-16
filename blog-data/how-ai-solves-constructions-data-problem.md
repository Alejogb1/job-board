---
title: "How AI Solves Construction's Data Problem"
date: "2024-11-16"
id: "how-ai-solves-constructions-data-problem"
---

yo dude so i just watched this killer presentation about this company trunk tools and their crazy ai for construction and lemme tell ya it blew my mind  it's basically this whole spiel about how they're using ai to fix the massive data mess in the construction industry and it's way more interesting than it sounds i swear

the whole point of the vid was to show how trunk tools is using ai to make construction way more efficient and less of a total nightmare  they're tackling this ridiculously huge problem of data mismanagement in construction projects think millions of pages of blueprints contracts change orders the whole shebang  it's like a digital swamp and they're building a digital dredge to clean it up

first off  the dude talking he's super enthusiastic which was awesome right away  he mentions this building in nyc  a skyscraper that used their software  that’s a huge visual cue he uses to showcase the scale of the data problem  he casually drops that they got access to 3.6 million pages of documents for just *one* building  can you believe that three point six million  that's insane  it's like he's saying "look at this mountain of paperwork we're dealing with"  another visual cue was this ridiculously huge spreadsheet he shows  like a three-foot-wide table crammed with numbers  totally unreadable for a human  it was like "yeah this is the problem we're solving"


then he gets into the meat of it  two major concepts really stand out  one is this idea of a "brain" for construction  they've built this system that basically ingests all these different file types  blueprints schedules emails  everything  and it all gets stored and processed in one place  it’s like a giant knowledge base specific to construction  think of it as a supercharged google search but for construction documents only

the second big idea is these "ai agents" they're basically little ai bots that live inside this "brain"  they can answer questions  they can even automatically create requests for information rfis which is a huge deal in construction because it saves tons of time and prevents mistakes  imagine you need to know if a specific door needs special hardware instead of sifting through millions of pages some ai agent just pops up the answer for you  bam

here's where it gets really techy dude. think about this python code snippet for a simplified version of their search functionality:

```python
import os
import re

def search_documents(query, doc_dir):
    results = []
    for filename in os.listdir(doc_dir):
        filepath = os.path.join(doc_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            if re.search(query, content, re.IGNORECASE):
                results.append({'filename': filename, 'content': content})
    return results

query = "power actuated hardware"
doc_dir = "/path/to/documents"
results = search_documents(query, doc_dir)
for result in results:
    print(f"Found in {result['filename']}: {result['content'][:100]}...")

```

this is a super simplified example but it shows how you'd search through a bunch of documents  this would need way more sophistication to handle different file types and to make sense of the often messy data  but it gets the point across right? it's all about efficiently searching and indexing that massive dataset


another snippet shows how you might create an rfi automatically if there's a discrepancy:

```python
def create_rfi(discrepancy, project_id):
    rfi_data = {
        'project_id': project_id,
        'description': discrepancy,
        'status': 'open',
        'assigned_to': 'engineer',
    }
    # send rfi_data to the rfi system
    print(f"RFI created for discrepancy: {discrepancy}")

discrepancy = "Door 213 requires power actuated hardware but docs say standard"
project_id = 12345
create_rfi(discrepancy, project_id)
```

see? this shows how their system could potentially detect inconsistencies and immediately generate a new rfi  avoiding the whole chain of communication breakdowns

and finally the kicker  this whole system isn't just about finding info  it's about fixing the problem  the presenter talks about how their agents can even create those rfis automatically  so instead of people wasting time figuring out conflicts  the ai does it for them  that's a total game changer  a third code snippet could be used here to show how different ai models could be used to classify the type of discrepancy


think of a natural language processing model:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidates = ["missing information", "contradiction", "design flaw", "material issue", "code violation"]

discrepancy = "Door 213 requires power actuated hardware but specs say standard"
results = classifier(discrepancy, candidates)

print(f"Discrepancy Classification: {results}")

#further processing here to determine course of action - create RFI, assign to engineer etc.
```
this is a simplified version but shows how NLP can categorize the type of discrepancy


the whole thing ends with the guy basically saying  "hey we're building this killer ai for construction  come join us"  it's all about streamlining the process by using ai agents to solve real world problems  this is what makes their product so revolutionary  not just creating another search engine but actually creating a proactive problem-solving tool


so yeah  that’s the gist of it  a super enthusiastic guy talking about how they’re using ai to rescue the construction industry from itself one rfi at a time and honestly after seeing that data mess i'm pretty much convinced that they're onto something really big  if you're into ai and construction you really gotta check out their stuff  seriously  it's pretty mind blowing
