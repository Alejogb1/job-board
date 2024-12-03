---
title: "How does ChatGPT's new chat search functionality improve the user experience?"
date: "2024-12-03"
id: "how-does-chatgpts-new-chat-search-functionality-improve-the-user-experience"
---

Hey so you wanna know about ChatGPT's new chat search thing and how it's all shiny and new user experience wise right  yeah me too its pretty cool actually  before this whole thing chatGPT was kinda like talking to a really smart parrot it knew a lot but it didn't really know where it got that info from  it was all kinda mashed together in its giant brain  you know what I mean  like asking it about the history of the Roman empire you'd get a decent answer but it wouldn't cite sources or anything it was just its own synthesis  a bit spooky honestly

But now with this search feature it's different  it's like it suddenly grew eyes and can look things up  it actually uses a search engine behind the scenes  I think it's Bing now but maybe Google or something else in the future  it's all very hush hush  the point is it can now actually verify its answers  and that's huge  it means you get more accurate information  less of that made-up stuff that used to happen sometimes its still possible of course but way less common


The user experience part is way better  it’s less guessing and more knowing  you ask a question it searches  then it gives you the answer  plus it shows you where it got that info from  this is key its like having a research assistant built right into the chat  I mean think about it  you don't have to go switch to Google or DuckDuckGo or whatever  you just ask your question and it does everything for you  its super convenient 


Before  if I wanted to know something specific I'd ask ChatGPT then I'd have to fact-check it myself  it was like a two-step process  now it's one  it's streamlined its faster its more efficient its less frustrating  I can get my info and move on with my life


For example say I want to know about the latest developments in quantum computing  before I’d ask ChatGPT and get some possibly outdated or even hallucinated answer  now I ask  and it actually searches and gives me results from current research papers  it'll even sometimes give me snippets from the papers themselves  I can even ask it to compare and contrast findings from different sources  its insane


The way it displays the information is also pretty slick  it's not just a wall of text  it's structured  it uses bullet points and summaries  it makes it easier to read and digest the info  it’s less overwhelming


Also it handles follow up questions way better  you can ask clarifying questions based on what it found  you can ask it to explain something in simpler terms or go deeper into a specific point  it’s like a dynamic conversation  not just a question-answer exchange   it's genuinely improved the conversational flow


Now here's where things get a bit more techy  I imagine they're using some kind of hybrid approach combining large language models with a robust search API  the LLM is still the brains it generates the text and answers questions  but the search API is its eyes it helps it ground itself in reality  they probably use some ranking algorithm to prioritize results too  giving more weight to authoritative sources  think about something like  PageRank but more sophisticated


The search results aren’t just displayed passively either the  LLM is actively processing and interpreting those results  this isn't just pasting links  it's actually synthesizing the info from multiple sources  giving you a coherent and concise answer  it's kinda like a sophisticated summarization model working behind the scenes  this probably involves techniques from information retrieval and natural language processing  you could look at papers on  document summarization and query expansion  also check out books on search engine technology


Let me give you some code examples to illustrate the kind of stuff I imagine going on behind the scenes  obviously I don't know the exact code but this gives you an idea


**Example 1: Search Query Processing**

This Python snippet shows how they might formulate a search query using relevant keywords from the user's input

```python
import re

def formulate_query(user_input):
    keywords = re.findall(r'\b\w{4,}\b', user_input.lower()) #Extract keywords (words with 4+ letters)
    query = ' '.join(keywords)
    return query

user_query = "What are the latest developments in quantum computing research?"
search_query = formulate_query(user_query)
print(search_query) # Output: what are latest developments quantum computing research
```

This is super simplified  a real system would use much more sophisticated NLP techniques like stemming lemmatization named entity recognition to refine the query  for more info on this check out books on Information Retrieval and NLP  Jurafsky and Martin’s Speech and Language Processing is a good one


**Example 2: Result Filtering and Ranking**

Here's how they might filter and rank search results based on relevance and authority

```python
results = [
    {'url': 'example.com/paper1', 'authority': 0.8, 'relevance': 0.9},
    {'url': 'example.com/paper2', 'authority': 0.6, 'relevance': 0.7},
    {'url': 'example.com/blogpost', 'authority': 0.3, 'relevance': 0.5},
]

def rank_results(results):
    return sorted(results, key=lambda x: x['authority'] * x['relevance'], reverse=True)

ranked_results = rank_results(results)
print(ranked_results)
```

In reality this would be way more complex  they’d use machine learning models to predict relevance and authority  you can look into papers on learning to rank and search result scoring for this


**Example 3:  Answer Synthesis**

Finally here's a basic idea of how they might synthesize an answer from multiple search results


```python
results = [
    {'title': 'Paper 1', 'text': 'Quantum computing is advancing rapidly.'},
    {'title': 'Paper 2', 'text': 'New algorithms are improving performance.'},
]

def synthesize_answer(results):
  summary = "Based on recent research"
  for result in results:
      summary += f" {result['text']}"
  return summary

answer = synthesize_answer(results)
print(answer) # Output: Based on recent research Quantum computing is advancing rapidly. New algorithms are improving performance.
```

Again very simplified  a real system would use advanced summarization techniques  and methods to avoid plagiarism  you could search for papers on extractive summarization and abstractive summarization


So yeah that's the gist of it  ChatGPT's new search function  it's a game changer for user experience  making it way more accurate reliable and convenient  behind the scenes it's a mix of NLP search algorithms and machine learning magic  its still evolving of course but it's already pretty impressive  I am stoked to see what’s next honestly
