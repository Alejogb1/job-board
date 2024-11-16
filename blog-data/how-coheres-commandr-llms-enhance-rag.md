---
title: "How Cohere's Commandr LLMs Enhance RAG"
date: "2024-11-16"
id: "how-coheres-commandr-llms-enhance-rag"
---

dude you are NOT gonna believe this video i just watched  it's like some crazy cool ai stuff from cohere  they're basically showing off their new llm family the commandr models and  man it’s a wild ride

so the whole point of the video is to show how they built these amazing models that are ridiculously good at this thing called rag which stands for retrieval augmented generation  basically imagine an ai that not only understands what you're asking but also knows where to find the answer and then gives you the answer with sources  it's way smarter than your average joe ai  they also talk a lot about how they've made it super easy for devs to use which is a huge deal

ok so key moments  first off the timeline thing they showed  it’s totally bonkers how quickly they released commandr and then commandr+  it was like boom boom  new model  it’s insane  they even mentioned hugging face loving their work so much they used it as a base for hugging chat  that's a seriously big deal

second this whole thing about prompt sensitivity  that's a HUGE problem with llms   they explain how hard it is to get the model to not only find info but also know where to look and how to separate the chat history from the new info  they're like yeah it's not easy  and honestly  i'm totally with them on that  i mean they call it an excruciatingly hard  tough word  and they are 100% correct

third  this model bias thing  it's totally relatable  llms often just grab the first thing they see and call it a day  cohere calls this the model's laziness and they've worked hard to fix it  they even mentioned needle in a haystack benchmarks which is a classic example of this problem

fourth citations  oh my god  citations  they're super pumped about making sure the models cite their sources  i totally get it   no more fake news from the ai  it's a total game changer  they're making it super fine-grained so you know exactly where the info came from  reducing hallucinations  which is basically ai making stuff up

fifth  the whole tool use thing  they've made this crazy-powerful multi-step functionality  imagine asking an ai to do a complex task like “find the latest weather forecast for london, compare it to the forecast from last week, and then email me a summary"  that's what they’re talking about  single-step is for simple things, but multi-step lets the model chain actions together  it even handles errors  like if an api call fails, it'll try again automatically  it’s magic

visual cues  i remember that timeline graphic they showed was awesome  really clean and simple  showed the whole development process clearly  also the code examples they briefly flashed  were seriously impressive  and the UI screenshots they showed  looked super slick  it's like the kind of ui i wish i could design  but let's be honest  i can't

now for some concepts  rag itself is pretty neat  it's this whole idea of letting the model grab info from external sources  like databases or websites  instead of just relying on what it learned during training this makes the ai much more versatile   here's a quick python example

```python
import langchain  # language chain library for rag, and many other awesome tasks

# a super simple example  you'll need to install the necessary libraries  like langchain, FAISS etc.
from langchain.llms import OpenAI  # or whatever llm you prefer
from langchain.vectorstores import FAISS  # FAISS library for efficient vector search
from langchain.chains import RetrievalQA

# this is a placeholder for your actual data that you would load and process
docs = ["this is the first document", "this is the second document, with more content."]

# using embedding to vectorize our data  replace OpenAIEmbeddings with other embedding techniques
embeddings = OpenAIEmbeddings()  # open ai's embedding technique
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
query = "What are the documents about?"
response = qa.run(query)
print(response)
```

this is just the barest bones  real-world applications are much more complex  but it shows the basic idea of using an external database to power the llm's knowledge

then there's this multi-step thing  that's even cooler  it’s all about letting the model perform a sequence of actions  think of it as giving the ai a plan  and it actually follows it  it's like writing a little program for the ai to execute   it’s much more complex than a simple one-step thing. here's a conceptual example which won't work without a proper implementation, using hypothetical functions:

```python
def search_documents(query):
  # simulates searching a document database
  return ["document1.txt", "document2.txt"]

def summarize_documents(documents):
    # simulates summarization
    return "This is a summary of the documents."


def send_email(summary):
    # simulates sending an email
    print("Email sent:", summary)

# multi-step workflow
user_request = "Compare weather forecasts for London last week and today, then email me a summary."
documents = search_documents("London weather forecasts")  # Step 1: Search
summary = summarize_documents(documents)  # Step 2: Summarize
send_email(summary)  # Step 3: Send email

```

this shows how you could break down a complex task into smaller, manageable steps.  the real magic is that cohere’s models automatically plan, execute, and adapt this kind of workflow

and finally, there’s a small bit of js that they showed off  this was for their open-source ui  they actually open sourced the whole thing  which is amazing  it's built with nextjs  and uses a small sql database  the cool part is  it can use the commandr models locally on your machine  or through other providers like hugging face  here's a little snippet to illustrate (again, this is highly simplified—real-world would be HUGE)

```javascript
// hypothetical component to display citations from the model's response
function CitationDisplay({citations}) {
  return (
    <div>
      <h3>Citations</h3>
      <ul>
        {citations.map((citation) => (
          <li key={citation}>{citation}</li>
        ))}
      </ul>
    </div>
  );
}
// the actual implementation would be complex and involve numerous components and state management
// with nextjs and sql handling
```

this part isn't super functional but shows how citations might be displayed.

the conclusion  is that cohere has built something seriously impressive  small models that are incredibly powerful  they focus on making rag accessible, robust, and reliable  with a clear focus on citations and a user-friendly interface  their commitment to open source is also a big plus  it's not just about building great ai, it's about enabling others to build with it, too  i'm seriously impressed, and i bet you are too!
