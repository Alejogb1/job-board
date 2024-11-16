---
title: "Building Reliable AI Apps with Guardrails"
date: "2024-11-16"
id: "building-reliable-ai-apps-with-guardrails"
---

dude so this shrea rajal chick gave this killer talk about building ai apps that don't totally suck  it was all about this "trust but verify" thing for generative ai which is like totally relevant now that everyone and their grandma's building autogpt stuff

the whole point of her shpiel was that  generative ai models like llms are super flaky  you know they hallucinate  they sometimes just make stuff up  and that's a huge problem if you're building anything that actually needs to be reliable like a medical diagnosis tool or a self-driving car or even just an app that doesn't randomly break


she showed this graph—i wish i could draw it but picture a rocket ship going straight up—it was charting the insane surge in interest for "artificial intelligence"  especially after chatgpt blew up  but then she dropped another bomb  a graph showing how fast adoption is versus how fast people stop using the thing  retention sucks!   like people try an ai app get excited then ditch it

one of the key moments was when she hammered home how different generative ai is from traditional software.  think about querying a database—every time you ask the same question you get the same answer, right?   consistent predictable  the whole foundation of everything we've built  but with llms it’s a crapshoot you ask the same thing five times and you might get five different answers.  pure chaos


another key takeaway was that the only tool you really have when dealing with llms is… english.  you're writing prompts.  that's it  no fancy compiler errors  no type checking  just words  and hoping the ai understands and behaves correctly


she dropped some sick visuals too like a diagram showing the typical ai app architecture  it's simple  app sends a prompt to the llm gets a response  but she proposed a new architecture with a "verification suite" sitting between the app and the llm  this suite acts like a bouncer only letting responses that pass various checks through


this verification suite is the heart of her "guardrails ai" framework she was pitching.  it’s open source woohoo  basically you write "guards"—functions that check the llm's output for issues.  like  is it hallucinating  does it contain profanity  is the code it generated actually executable  etc.  these guards let you add correctness guarantees without killing the flexibility of the llms



let’s talk code.  imagine you're building a chatbot that needs to answer questions based on a set of articles. you wouldn't want it making things up right


```python
import guardrails  # importing the guardrails library

# defining a guard using Guardrails's declarative config (for example in YAML)
guard_config = """
name: my_article_guard
checks:
  - type: provenance
    source: articles.json #  your articles data
  - type: profanity
    threshold: 0.8 #  detect profanity
  - type: competitor_mention # checks for mentions of competitors
    competitors: ["competitorA", "competitorB"]
"""

guard = guardrails.Guard.from_yaml(guard_config)


def get_answer(question, context):
    prompt = f"Answer the question based on the context:\nQuestion: {question}\nContext: {context}"
    response = guard.run(lambda: llm_call(prompt)) # the llm_call function handles actual LLM interaction
    return response["output"]


def llm_call(prompt):
    # replace this with your actual LLM call
    # this is a placeholder for any llm function
    # e.g., openai.Completion.create
    # or a custom function calling your llm
    #  ...some complex llm call using some library...
    return {"output": "some response from the llm"} # replace this with actual llm response

question = "what's the best way to treat a headache"
context = "Some medical article content about headaches and treatments"
answer = get_answer(question, context)
print(answer)
```


this code uses guardrails to ensure the answer is grounded in the provided medical context  doesn't contain profanity and avoids mentions of competitors. the  `guard.run` wraps the actual llm call so that all checks run automatically before the response is returned.


another example  say you're generating sql queries.  you absolutely want to avoid sql injection vulnerabilities


```python
import guardrails
import sqlite3

guard_config = """
name: sql_query_guard
checks:
  - type: sql_injection
  - type: executable_query
    database: mydatabase.db #  path to database
"""
guard = guardrails.Guard.from_yaml(guard_config)

def generate_sql(prompt):
    #some function that generates sql using llm
    #  ...llm call to get sql query...
    sql_query = guard.run(lambda: llm_call(prompt)) # executes llm call and checks it
    return sql_query["output"]


def execute_query(query):
    conn = sqlite3.connect('mydatabase.db')
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        conn.commit()
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        conn.close()
        return None

user_prompt = "give me sql to select all users where name is 'bob'"
sql = generate_sql(user_prompt)
if sql:
  results = execute_query(sql)
  print(results)
```

here guardrails makes sure the generated sql is safe and executable against a real database.  it's not just syntax checking. it's checking if the query actually does what you intend and doesn't allow anything dangerous



and a final code snippet to illustrate the "re-prompting" loop. if a check fails guardrails automatically tries to fix it.


```python
import guardrails

guard_config = """
name: my_reprompt_guard
checks:
  - type: fact_verification
    source: wikipedia # you can use any source
policies:
  - type: rephrase
    retries: 2 # two retries before giving up
"""
guard = guardrails.Guard.from_yaml(guard_config)


def get_fact(prompt):
    result = guard.run(lambda: llm_call(prompt))
    return result

fact_query = "who invented the lightbulb"
fact = get_fact(fact_query)
print(fact)
```

in this case if the llm gets the fact wrong  guardrails will automatically re-prompt it with more info for improved accuracy.  this is where the "trust but verify" loop shines


so yeah that was shrea's talk  it was pretty mind-blowing  the whole guardrails ai framework is clever  it's basically adding a safety net to generative ai making it much more practical for building real-world applications  instead of just cool demos.  she made a good point about how just prompt engineering isn't enough llms are still gonna act dumb sometimes no matter how cleverly you craft your prompts   you need a systematic way to handle these inevitable failures and guardrails seems to provide that
