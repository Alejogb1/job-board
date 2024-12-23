---
title: "How to Use Hyperspace AI's Decentralized Network"
date: "2024-11-16"
id: "how-to-use-hyperspace-ais-decentralized-network"
---

dude so i just saw this killer demo for hyperspace ai and lemme tell ya it's wild  it's basically this decentralized ai network built on people's computers, like a crazy distributed computing thing but for ai  nicholas schlapfer the guy in the video is all like "yo we don't have a mountain of gpus we use your home computer's power" which is pretty freakin' awesome right  the whole point is to create this amazing ai experience using a bunch of different models instead of just one mega-corp monster model.  think of it like a massive ai jam session instead of a single orchestra playing the same boring symphony over and over


so the big deal is this new thing they're calling hyperspace it’s their flagship product built on this distributed network they already have which is called aios—that uses llama.cpp for inference and they’re already rocking it on windows and mac  it’s nuts because you could be chatting with an ai powered by someone's computer in belgium  thats next level distributed computing if i ever saw it


five key moments that blew my mind:


1. **decentralized power**:  the whole decentralized thing is huge  no massive data centers hogging all the resources  it’s all about harnessing the collective power of home computers and the visual of that decentralized network working was a pretty sick graphic they had in the video  it’s like a visual representation of all the little nodes connected in a giant web, each contributing to the total processing power. it felt like that time i saw that documentary on ants—but with way more computing power and less creepy crawlies.


2. **the dag orchestration**:  this is where things get really interesting they’re using a directed acyclic graph (dag) to orchestrate the whole ai process.  a dag is basically a flowchart that defines the order of tasks.  so, you give it a query, and it generates a dag  this dag outlines all the steps needed to answer your question.  and then, get this each node in the dag represents a specific task, maybe scraping a website, running some python code, or querying another model.  it’s like a super-organized way to break down complex problems into smaller, manageable chunks


3. **the node editor**:  this is hands down the coolest part they built a node editor think of it like a visual programming environment where each node in the dag is represented by a box and you can connect them, change their parameters, and even add new nodes  this lets you fine-tune the whole process for any specific query. the video shows him actually dragging these little boxes around like it's some super fun puzzle game, and that's just way cooler than writing endless lines of code.


4. **web scraping and llm integration**:  they use puppeteer and beautiful soup for web scraping  puppeteer is like a headless chrome browser, allowing it to interact with websites automatically  beautiful soup then parses the html making it easy for the llms to understand.  think of it like a highly trained robotic librarian who can locate any information, and then explain it to you in a way you actually understand, instead of in legalese and 10-point font.


5. **python execution inside the node editor**:  the ability to execute python code directly within the node editor is a game-changer  this adds a ton of flexibility and it lets users customize each step in the process.  imagine you need to do some complex data manipulation that requires a specific python script —boom you just insert a python node into your dag and it all happens automatically, within the workflow.


 now let's talk code examples.  i’m not gonna write full implementations but i'll give you the flavor:


example 1:  puppeteer web scraping (simplified):


```python
async def scrape_website(url):
    browser = await launch()  # launch headless chrome
    page = await browser.newPage()
    await page.goto(url)
    html = await page.content() # this gets the html of the page
    await browser.close()
    #beautiful soup is not shown but you'd parse the html here
    #return processed_data
    return html

#example usage
scraped_data = await scrape_website("https://www.example.com")
print(scraped_data)

```

this is a super basic example it’s just to show you how puppeteer would pull down the html from a site.  in reality, you’d need error handling, more sophisticated parsing, and much more but you get the idea  puppeteer handles the browser interaction and you just get the raw html back.  beautiful soup would then handle the cleaning of that html into a usable format.


example 2:  a simple dag node (conceptual):


```python
class DagNode:
    def __init__(self, task_name, function, inputs=None, outputs=None):
        self.task_name = task_name
        self.function = function  # the function this node executes
        self.inputs = inputs or []  # list of input nodes or data
        self.outputs = outputs or []  # list of output nodes or data

    def execute(self):
        # fetch inputs
        # execute function
        # produce output to other nodes
        pass


#Example Usage
def add_one(x):
    return x + 1

node1 = DagNode("add_one", add_one) # A node that only does one thing
node1.execute(5)  #returns 6

```

this is just a basic class to represent a node  the `execute` function would contain the logic for that node. in hyperspace, each node could be anything from a simple mathematical operation to a complex llm query.  the inputs and outputs would be the data flowing between nodes based on the dag structure. this is why they have that node editor, because each node would need to be able to accept inputs, process them, and produce outputs in a structured manner.


example 3:  llama 2 interaction (again, super simplified):


```python
import openai #you'd need to set your API key here
def query_llama(prompt):
    response = openai.Completion.create(
        model="text-davinci-003", #or whatever model they're using
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# example usage
answer = query_llama("what is the meaning of life?")
print(answer)
```

this is just showing you a simple interaction with an llm  the hyperspace system uses multiple llms, which adds a whole layer of complexity not shown here  but this captures the essence of how a node might use an llm to get an answer then you can combine that answer with the results from other nodes like the web scraper.  they’re using llama 2 or some other big model for the actual core ai responses


so the resolution? hyperspace is a really ambitious project  it aims to democratize access to advanced ai by decentralizing the computing power needed  the combination of dag orchestration, the node editor, and the integration of various ai models and python execution is pretty innovative. i'm really curious to see where this project goes it's on my radar and i’ll be trying it out as soon as it’s available. it is a really fun demo to watch and the guy in the video seems super passionate about what they're building.  i can’t wait to mess around with that node editor myself.
