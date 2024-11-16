---
title: "Building AI Agents with Langchain: A Practical Guide"
date: "2024-11-16"
id: "building-ai-agents-with-langchain-a-practical-guide"
---

yo dude so i just finished watching this killer workshop on building ai agents and man it was a trip lemme break it down for ya in the most casual way possible


the whole shebang was about getting you up to speed on building these things called ai agents  think of them as little robots that use those fancy large language models llms to solve problems not just answer questions  we're talking multi-step missions here not just simple q&a  the presenter aura  a total rockstar btw spent like 20-30 mins laying the groundwork then the rest of the time was hands-on  building our own agent with help from her team which was pretty sweet


one thing that immediately caught my eye was aura's intro  she's a dev advocate at mongodb  previously a data scientist in cybersecurity  she’s got the cred  she also mentioned yoga and coffee shops so major respect


key moments right off the bat were these 


first aura pointed out the diff between simple prompting rag (retrieval augmented generation) and agents   simple prompting is like asking a question directly to the llm  it’s fine for basic stuff but struggles with complex tasks  rag is better it lets the llm access a knowledge base so it can answer more detailed questions  but still it lacks the multi-step problem-solving power of agents


then she hammered home the agent’s three main components planning and reasoning memory and tools  


planning and reasoning  was a big deal  she showed us two styles chain of thought and tree of thought  chain of thought is just that think step-by-step  it’s straightforward but tree of thought takes it to the next level  it explores multiple paths like a decision tree  then she showcased  react and reflection react is about the llm doing something observing the results and refining its plan dynamically reflection is similar but the llm steps back and analyzes its own actions before proceeding


memory was interesting  she talked about short-term and long-term memory  short-term is pretty easy just storing the current conversation  long-term is tricky storing and updating information across multiple interactions which is crucial for personalized responses  this part is still kinda new territory


lastly there were tools  these are how the agent interacts with the outside world  they could be anything from simple apis like a weather api or something complex like a vector database or a custom ml model  it’s super important that your llm knows what tools are available and how to use them



code snippets were scattered throughout  we used langchain quite a bit which is a super handy library for this stuff here’s a little something i remember from building one of the tools



```python
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load a document from a URL
loader = UnstructuredURLLoader(urls=["https://example.com/research-paper.pdf"])
documents = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# Query the database
query = "what is the main topic of this paper?"
results = db.similarity_search(query)

for doc in results:
  print(doc.page_content)
```

this shows a simple example of processing a research paper using langchain  we load a document split it into chunks create embeddings and then query the vector database to answer questions


another snippet i recall was about using the langchain expression language lc it’s  for defining workflows   


```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Define the LLM
llm = OpenAI(temperature=0)

# Define the prompt template
template = """Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
question = "What is the capital of France?"
answer = chain.run(question)
print(answer)
```

this is a basic example  you define your llm a prompt template and create a chain  the chain executes when you provide input


finally we had this bit where they showed us how to build a basic agent in langchain using tools  


```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("What was the high temperature in SF yesterday and what is that temperature in Celsius")

```

here we load tools  initialize an agent and run a query  the agent figures out which tools to use based on the query  pretty slick


the workshop ended with building a research agent using fireworks ai’s model mongodb as a vector store and langchain's tools  it was designed to answer research-related questions find papers and summarize them  the whole thing used short-term memory  it remembered previous interactions within a session


so yeah that's the lowdown on the workshop man  super insightful and a ton of fun   it really clarified how agents work the different reasoning techniques and how to use langchain to build them  definitely gonna try to build some cool stuff now  peace out
