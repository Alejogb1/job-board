---
title: "Analyzing the Current State of LLMs"
date: "2024-11-16"
id: "analyzing-the-current-state-of-llms"
---

hey dude so i just watched this killer talk about the current state of llms and man it was a trip  it was this guy simon wilson and he basically laid down the smackdown on everything that's happening in the world of large language models right now  the whole thing was like a rollercoaster of insights humor and code snippets so buckle up buttercup because i'm about to break it all down for you

the setup was this: simon's main point was that the "gpt-4 barrier" the idea that gpt-4 was untouchable in terms of performance has been totally smashed to smithereens  he wanted to show how other models have caught up and even surpassed gpt-4 in certain areas and then he also dives into some serious ethical and usability issues  like a whole lotta fun stuff

**key moment 1: the gpt-4 barrier is GONE**

remember when gpt-4 dropped and everyone was like "holy moly this thing is amazing" yeah well simon showed how that's no longer the case  he used this awesome chart (which he totally re-created using gpt-4's code interpreter because the original was outdated lol) that plotted model performance against cost  the chart had these three distinct clusters:

* **the elite:**  gpt-4-0 gemini 1.5 pro and claude 3.5 sonet these bad boys are top-tier and relatively cheap
* **the budget beasts:**  claude 3  gemini 1.5 flash models these are super affordable and still really powerful  great if you're building something and cost is a factor
* **the dumpster fire:** gpt-3.5 turbo  simon was totally brutal saying it's now pretty much garbage  move on dude

he mentioned this mml benchmark  which is like a super hard bar trivia quiz for llms which is funny because it doesn't really reflect real-world usability  but hey it's the standard so everyone uses it

then he brought up lm-eval-harness which is an open-source framework that lets you evaluate language models  it's like having a really detailed report card for each model


**key moment 2: vibes matter more than benchmarks**

simon is not wrong about this llm-eval-harness is great but he pointed out that  pure benchmark scores don't tell the whole story  the real test is how well these models perform in real-world tasks and how they "feel" to use  he introduced the lm-versus-chatbot-arena this is basically a giant showdown where random users vote on which model is better at a given task  its an elo ranking system so its like the ultimate popularity contest for llms  and guess what  openai's gpt-4 is still up there  but claude is right behind it  and even some open-source models like llama 3 are making serious inroads  this whole thing's super dynamic with new models and rankings always changing  this arena is a must-see for llm enthusiasts


**key moment 3: open source is killing it**

seriously man open source is rising up  simon showed how models like llama 3 are now competitive with gpt-4 in terms of performance but they’re also open source and can run on a normal computer  this is a huge deal because it means that the power of llms is becoming democratized which is super radical


**key moment 4:  the usability nightmare**

simon had some choice words for the usability of chatgpt  he asked a very simple question "under what circumstances is it effective to upload a pdf to chatgpt" he admitted he didn't even know the answer  and then he went on a whole rant about the hidden complexities  the pdf has to be searchable not just a scan  short ones get pasted long ones get indexed somehow but who knows how tables and images are a nightmare  sometimes it even uses code interpreter with these mysterious modules like fpdf  pdf2image  and pypdf  he's even scraping code interpreter's package list using github actions to figure out what's going on because openai doesn't provide clear documentation  crazy right  this whole section highlighted that even something as seemingly simple as chatgpt has a lot of hidden quirks  making it much harder to use effectively than it first appears  it's like excel  anyone can use it but mastering it takes years


here's a little python code snippet that mirrors his github action idea:

```python
import requests
import json

# this is a placeholder  you need to find the actual api endpoint for code interpreter's package list
api_url = "https://example.com/code_interpreter_packages"

response = requests.get(api_url)
data = json.loads(response.text)

with open("code_interpreter_packages.json", "w") as f:
    json.dump(data, f, indent=4)


print("package list saved to code_interpreter_packages.json")
```

this code fetches a json file (the package list) and writes it to a local file  of course it will need the correct `api_url`


**key moment 5: the ai trust crisis**

the other big issue simon talked about was the growing lack of trust in ai companies  he gave examples of dropbox and slack both getting slammed for seemingly training models on user data even though they weren’t  this was due to poorly worded terms of service and default opt-in settings  the trust problem is huge  and the solution is transparency  simon praised anthropic's claude model for explicitly stating that they didn't use user data for training  but even then the original sin remains:  they trained on a huge unlicensed web scrape


here's another code snippet to illustrate some of the security concerns:


```python
# simulated markdown rendering  very simplified
markdown = "![image](?q=my_secret_api_key)"
# in a real system this would invoke a markdown renderer and possibly access the external resource


# insecure markdown rendering would lead to data exfiltration
# a secure implementation would either block or sanitize this kind of input
if "?" in markdown and "q=" in markdown:
    print("potential data exfiltration detected")
    # in a real system this would raise an alert or trigger a mitigation strategy
```

the insecurity in the code is obvious but many companies are making the same mistakes


here's a last code example that illustrates a simple prompt injection vulnerability

```python
#  a vulnerable function that processes user input and uses it in a system command
def process_user_input(user_input):
    command = f"echo '{user_input}'" # very vulnerable! always sanitize user input
    os.system(command)


user_input = input("enter some text: ")
process_user_input(user_input)

# this is EXTREMELY insecure  an attacker could craft malicious input to execute arbitrary commands

```

this is why prompt injection is such a huge deal



the resolution is simple: we're in a golden age of llms  but it's up to us the developers and users to use them responsibly and ethically  we need to understand the strengths and weaknesses of these models and to build tools that are both powerful and safe  plus we gotta fix this usability problem because right now they're only truly usable by power users

so there you have it  simon wilson's talk was packed full of insights and jokes  it was an excellent overview of what's going on and the challenges that lie ahead in the wild world of llms   hopefully this breakdown helped  peace out dude
