---
title: "How to Connect AI to Your Data Using SQL and LLM"
date: "2024-11-16"
id: "how-to-connect-ai-to-your-data-using-sql-and-llm"
---

dude so i just watched this amazing talk—this guy's totally onto something  it's all about connecting your ai to your data like seriously making them best buds  the whole point is that right now ais are ridiculously good at some things—like making flappy bird with your face in it in 30 seconds—but they're total dumbos when it comes to actually understanding and using your real-world data  think your salesforce data your calendar your freakin' amazon order history   it's nuts

he starts off with this hilarious rant about how frustrating it is that he can ask an llm to make a silly game but he can’t get it to answer something simple like "how many one on ones did i have last week"   it's like these superbrains are stuck in the toy box  he gives the example of trying to get info about an amazon delivery  it's 2024 and it's still a major pain to check if something can be delivered to your address it’s  a 'death by a thousand cuts' scenario  the whole thing is way too complicated and the security aspect is a nightmare

one thing that stood out was this part where he's like "llms don't know what your API does"  that’s so true most people don't even fully know what their own apis are doing  but they understand sql  that's his first major idea—make everything talk the same language  use sql as a common interface even for unstructured data  think of it like a universal translator for data  apis databases spreadsheets—all speaking the same sql dialect  

it was pretty funny when he called himself a "blockbuster" and used that as an example  he wanted to write an email to his top customers using the blockbuster data and this is the type of thing you'd want to have done with an ai   so he set up a system that does this:

```python
# simplified example  imagine a more complex system pulling from multiple sources

import sqlite3 # or whatever database you use
import smtplib # for sending email

# connect to the database
conn = sqlite3.connect('blockbuster.db')
cursor = conn.cursor()

# fetch customer data (replace with actual query)
cursor.execute("SELECT name, last_watched FROM customers WHERE customer_id=1")
customer_data = cursor.fetchone()

name = customer_data[0]
last_watched = customer_data[1]

# construct email
email_body = f"""
hi {name},

thanks for being a loyal blockbuster customer!

we noticed you recently watched "{last_watched}"—great choice!  

[more email content]
"""

# send email (replace with your credentials)
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("your_email@gmail.com", "your_password")
server.sendmail("your_email@gmail.com", "customer_email@example.com", email_body)
server.quit()

conn.close()

```

this is a tiny bit of code but it shows the basic idea  it pulls customer data using sql constructs it into an email and sends it using an email library  it's basic but it shows how sql acts as a common language  

his second point was about authorization this guy's a genius  his idea is to create an object model for authorization  instead of having crazy complicated rules for every single data source  he proposes that each piece of data has a schema and each user session has its own properties  then you just run a set of authorization rules and it works the same way no matter where the data is from

```python
# conceptual authorization model

class DataObject:
    def __init__(self, schema, data):
        self.schema = schema
        self.data = data

    def is_authorized(self, session):
        # check rules based on schema and session
        for rule in authorization_rules:
            if not rule.check(self.schema, self.data, session):
                return False
        return True


# example rules
class Rule:
    def check(self, schema, data, session):
        # implement your logic here
        pass

# example implementation  replace with your rules
authorization_rules = [
    Rule(),
    Rule(),
    # add more rules here
]

data = DataObject({"field1":"int","field2":"string"}, {"field1":10,"field2":"hello"})
session = {"user_id":1, "role":"admin"}
if data.is_authorized(session):
    print("access granted")
else:
    print("access denied")
```

this shows a simplified approach  authorization would become more complex but the idea is to centralize it to deal with many data sources  

his third idea—and this is where things get wild—is getting the llm to figure out how to get the data itself  this is insane he’s like "don't hardcode anything" let the llm write the python code  

```python
# this would be part of a more complex system

def get_data_from_llm(query, data_sources):
    prompt = f"""
    write python code to answer this question using the specified data sources:

    question: {query}

    data sources: {data_sources}

    the code should:
      1. connect to each data source
      2. execute queries
      3. return the results in json format
    """

    # use an llm to generate the code
    code = llm_api.generate_code(prompt)
    
    # execute the generated code
    # (caution - sanitize generated code before execution)
    result = run_code(code)
    return result
```

that's the basic gist  you give the llm a query and info on where the data lives  it then generates the python code to fetch it which is super cool

so the whole thing wraps up with this idea of a "data singularity"  it's like he's saying if we're heading toward an ai singularity we're also heading toward a data singularity  we need to build systems that let ais and data talk to each other seamlessly  it's not just about fancy algorithms  it's about making the data easily accessible  

it was a great talk lots of good laughs  but mostly it was insightful in showing a really important connection between AI and data and how that connection can improve greatly
