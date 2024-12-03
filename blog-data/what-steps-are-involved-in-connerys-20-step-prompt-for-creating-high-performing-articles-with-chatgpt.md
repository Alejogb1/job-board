---
title: "What steps are involved in Connery’s 20-step prompt for creating high-performing articles with ChatGPT?"
date: "2024-12-03"
id: "what-steps-are-involved-in-connerys-20-step-prompt-for-creating-high-performing-articles-with-chatgpt"
---

Okay so Connery's 20-step thing right its pretty intense but I've been messing around with it and it's actually kinda cool  I'm not gonna list all 20 steps verbatim because that's boring but I'll give you the gist and some coding examples to show you what's up

First off its all about iterative refinement you don't just chuck a prompt at ChatGPT and expect gold  It's more like sculpting think of it like you're working with clay you gotta keep shaping and reshaping until you get the thing just right  Connery emphasizes this a lot


Step 1 is all about defining your goals  What exactly do you want from this article  Do you need a funny story an informative piece a persuasive argument you gotta be crystal clear


Step 2 is about knowing your audience who are you writing for this is huge  Tailor your language and style accordingly  Imagine writing a tech blog post versus a heartfelt letter to your grandma


Steps 3-5 are about brainstorming keywords  researching existing content and outlining a basic structure  You want to find out what's already out there so you can make yours better and stand out  Think about competitor analysis but for articles


Step 6-8 are where things get interesting  This is where you start crafting your initial prompt  Connery stresses simplicity at first  Don't try to be too fancy  Just get a basic article going  We're talking bare-bones here


Here's where a code example helps

```python
initial_prompt = "Write a short article about the benefits of using Python for data science"
```

This is about as basic as it gets  You'll use this prompt to get a first draft to see where you're heading  Think of this as "version 1"  It might be crap but it's a starting point  Check out a book on prompt engineering for more detail on structuring prompts


Next few steps 9-12 are all about analyzing that first draft  What's good  What's bad  What needs improvement  This is where you start making changes  Adding details  Improving the flow  Fixing mistakes


Let's say the first draft is kinda weak on explaining specific Python libraries for data analysis  You might then refine your prompt like this


```python
refined_prompt = "Write a short article about the benefits of using Python for data science focusing on the Pandas NumPy and Scikit-learn libraries  Provide code examples for each"
```

See how we added specifics  This is key to getting better results  You could look at research papers on natural language generation to get better ideas on how to craft prompts effectively


Steps 13-16 are about iteration  You're going to run this refined prompt and then analyze the result again  And again  And again  It's iterative  This is where you really start to hone your article  Think about it like agile software development  You're constantly testing and improving


Here's a Python snippet that simulates iterative refinement you could imagine this as a loop that keeps running


```python
iterations = 5  #number of iterations
prompt = initial_prompt

for i in range(iterations):
    response = chatgpt_api(prompt) #function to call chatgpt api
    analysis = analyze_response(response) #function to analyze output
    prompt = refine_prompt(prompt, analysis) #refines prompt
    print(f"Iteration {i+1}: Prompt = {prompt}, Response = {response}")

```

This is a simplified representation and you would need to implement the `chatgpt_api`, `analyze_response`, and `refine_prompt` functions but it illustrates the idea  This process can be automated to some degree  Look into papers on automated prompt engineering techniques


Steps 17-19 are more about polishing  Fact-checking  Adding visuals  Optimizing for SEO and readability  You know  Making it look pretty and making sure its actually good


Step 20 is about publishing and promoting your work  This is where you put it out there and let people see it


Overall Connery's method is all about the process its not about some magic formula  Its about understanding ChatGPT's strengths and weaknesses learning to work with it  iterating refining  and constantly improving  Don't expect perfection on the first try  It's a process  Its a conversation  You're collaborating with an AI  The results will improve as you improve your skills


You should look into some resources to go deeper  Search for papers on "Large Language Model Prompt Engineering"  There's some really interesting stuff out there  Also  books on "Effective Writing" and "Technical Writing" are going to be invaluable for this process   You might also search for research on "human-in-the-loop" machine learning  its about how humans and AI work together to achieve something  It's pretty relevant to this method


The most important thing to remember is that this isn't just about using ChatGPT  It's about developing your own writing skills and understanding the principles of crafting compelling content  ChatGPT is just a tool its a powerful tool but it still needs a skilled operator  Think of it like a really sophisticated word processor its awesome but you still gotta know how to write
