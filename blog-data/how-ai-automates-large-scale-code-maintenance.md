---
title: "How AI Automates Large-Scale Code Maintenance"
date: "2024-11-16"
id: "how-ai-automates-large-scale-code-maintenance"
---

dude so i watched this talk about grit and man it was wild  the whole thing is about how they're using ai to seriously level up code maintenance and generation at a scale most companies can only dream of it's not about building whole new apps from scratch which is what most ai demos do its about taking existing massive codebases and making them way better faster basically making engineers superhuman

the whole setup is this guy  rantee founder of grit  he's got a google cloud background so he's seen firsthand how much time companies waste on tedious code updates he starts by saying most customer requests weren’t for shiny new apps but for ways to make their existing million-dollar systems run smoothly on kubernetes that's a huge pain point and it’s why grit exists to automate the boring parts

one of the first things that pops out is how he contrasts different kinds of ai developer tools there are the id assistants which are like autocomplete super useful but only work on tiny bits of code at a time then there are the "lower the floor" ai agents that lets non-programmers build apps these are cool in theory but he thinks this is pretty unrealistic for the vast majority of enterprise software that's where grit comes in grit's job is to "raise the ceiling" for senior engineers to be able to do ten times the work that they are already doing

he talks about this massive customer with thousands of repos and thousands of developers they wanted to switch to open telemetry instead of their old logging system which usually means a huge effort months of coordination  lots of meetings a sprawling excel sheet to track everything which he jokes is grit's main competitor  but grit did it in a week that's the power of one engineer using grit to manage multiple ai agents to churn out hundreds of pull requests across thousands of repos its insane

 so key idea number one is how grit uses a multi-step process for these massive code changes  first they index the whole codebase  not just looking at words but understanding the code's meaning using semantic indexing and more traditional static analysis to map dependencies and code structure like understanding the function calls and their imports and making sure they're all working together

then they use large language models and sub-agents to make changes  the cool part is they use something called gql a custom query engine that combines semantic search with a deep understanding of the code structure it's way more powerful than just throwing everything into a large language model and hoping for the best because it can actually find specific places to make changes even across a massive repo

key idea number two is how they address the unreliability of llms  even the most powerful models still make mistakes which is why grit uses the compiler  that's right  tsc if it's typescript  or equivalent for other languages grit runs the code gets compiler errors and feeds those errors back to the llm which helps the model understand and fix its own mistakes he explicitly calls compilers "rock"

a major point he makes is how slow compiling even a small change can be on really large codebases  it can take 10 minutes just to type check one part of a large application  this completely kills the workflow if you have to wait that long after every little change in the code so grit uses in-memory indexing and a bunch of optimization tricks like reusing parts of the compilation result to speed things up

to make things even faster grit takes snapshots of the in-memory state using firecracker a vm manager  this allows them to fork the environment to try multiple changes in parallel  they run multiple agents side-by-side testing changes and then use a voting mechanism to choose the best result he jokes that this is like a distributed database not an agent

also grit is really smart about making edits it avoids generating whole files again which is computationally expensive  instead it uses something called gql to do a precise search and replace using a before-and-after code snippet its much more efficient than trying to generate a giant diff

the resolution is basically grit makes large-scale code changes and maintenance feasible  it allows teams to make changes that would have been impossible before because of the time and coordination required  the whole system is built to work around the limitations of llms and to take advantage of existing developer tools like compilers and language servers

here are some code snippets to illustrate  these are simplified obviously

snippet 1: gql query

```python
# simplified gql query to find log statements
query = """
  find function_call(name: "log", args: { contains: "error" })
  where imported_from: "log4j"
"""

#  in reality this query would be much more complex
#  involving  semantic analysis and potentially traversal of the import graph
# to avoid false positives


results = gql_execute(query, codebase) # hypothetical gql_execute function
print(results) # returns list of locations to modify
```

this is super simplified but shows the idea of using a specialized query language to find specific code sections instead of just relying on generic semantic search

snippet 2: compiler-assisted llm improvement

```typescript
//example of broken code generated by llm
let gritRange: {start:number, end: number} =  { start: 10, end: 20} //incorrect type

//after compiler error and llm correction
let gritRange: LSP.Range = convertLSPRangeToGritRange({ start: 10, end: 20}); //correct type with additional context
```

this highlights how grit uses type checking to catch errors and then gives the llm a chance to fix them


snippet 3:  efficient code modification using a "before/after" snippet

```python
# simplified example of before/after code snippet for gql search and replace

before_snippet = """
  logger.error("something went wrong: {0}", error);
"""

after_snippet = """
  opentelemetry.trace.error("something went wrong: {0}", error);
"""


# grit would use this to replace all instances of before_snippet with after_snippet
#  using the gql engine for efficient and precise matching
# allowing it to handle variations in code style while avoiding false positives

```

this shows the power of using a structured before-after approach rather than generating whole files its way faster and less prone to errors

so yeah grit's a really cool tool it's not just about building new things with ai its about making the day-to-day lives of engineers drastically better by automating the tedious and time consuming tasks  its smart use of llms along with existing developer tools and efficient optimization techniques  is what makes it so powerful  i'm pretty impressed
