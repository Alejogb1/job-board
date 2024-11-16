---
title: "Building Production-Ready RAG Apps: A Practical Guide"
date: "2024-11-16"
id: "building-production-ready-rag-apps-a-practical-guide"
---

dude so jerry from lindex gave this killer talk on building rad apps and it was like a total brain dump of awesomeness but also kinda hilarious at points especially the bucket hat raffle part lmao

the whole point was to show how to make rag apps actually work in the real world not just some theoretical bs  he spent like a solid 20 minutes just laying out the problems that make production rag a nightmare  think about it  you're trying to get a language model to answer questions using information it's never seen before that's a recipe for disaster if you don't know what you're doing

he started by casually mentioning the two main ways to make llms understand your data  retrieval augmentation which is like shoving context into the prompt and fine tuning where you literally change the model itself  he mostly focused on retrieval augmentation though  which is honestly where most of the chaos lives

one super memorable visual cue was this slide showing the “rag stack” two parts data ingestion and data querying which includes retrieval and synthesis  imagine it like a burger  data ingestion is the bun data querying is the patty retrieval is the cheese and synthesis is the special sauce  you can't have a good burger without all of them

another great point he hit was the current state of things  he called it "naive rag"  it's like the first attempt at building a rag app it’s simple but super prone to errors  he literally pointed out how bad the response quality can be  things like hallucination where the model makes stuff up and low recall meaning it misses important information   it's like trying to make a perfect burger with rotten ingredients  you're doomed from the start

the third thing i really dug was his explanation of evaluation  he stressed how crucial it is to have a way to measure performance  it’s not enough to just build something  you need to know if it's actually working and you need data  lots and lots of data   it's all about setting a benchmark so you can improve your system  he talked about retrieval metrics like success rate and ndcg these sound complicated but they’re basically ways of measuring how good your search results are  i mean he was serious about this part he even mentioned a workshop just dedicated to this topic  

ok so now for the juicy bits  the techniques  he talked about a bunch but three stood out to me

first was chunk size optimization this is like the goldilocks problem of rag  too small and you lose context too big and the llm gets overwhelmed he made it clear that more tokens don’t always mean better performance  it's all about finding that sweet spot for your specific data this is where the code comes in

```python
# simple example of chunking text
from llama_index import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
from llama_index import LLMPredictor, PromptHelper

# load documents
documents = SimpleDirectoryReader('./docs').load_data()

# set chunk size and other parameters
chunk_size = 512 # this parameter can be tuned
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# create llm predictor
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

# create index
index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=chunk_size)

# query the index
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic of this document?")
print(response)
```

this code shows how to adjust the `chunk_size` parameter in llama index to control how your text is split before embedding and querying which directly relates to what jerry was discussing you can play with that number to see what works best

second was metadata filtering this was like a mind-blowing concept  imagine adding extra structured information to your data like labels keywords and even summaries  this lets you refine your search way beyond simple keyword matching it's like adding a supercharged filter to your vector database  jerry used the example of searching a 10k document  filtering by year makes the search much more precise here's some pseudo-code illustrating the concept


```python
# pseudo-code for metadata filtering
query = "what are the risk factors in 2021"
metadata_filter = {"year": 2021}  # adding a metadata filter

# vector database query with the filter
results = vector_db.query(query, metadata_filter)

# process results
# ...
```


third  was the idea of “small to big” retrieval  instead of embedding huge chunks of text he suggested embedding smaller bits like sentences then expanding the context during synthesis  this increases precision because the smaller chunks are more likely to be relevant but the llm still has access to all the info it needs to answer the question properly


```python
#Conceptual code illustrating small to big retrieval.  Actual implementation highly library-specific
def small_to_big_retrieval(query, small_chunks, large_chunks, retriever):
  initial_results = retriever.search(query, small_chunks) #search smaller chunks first
  relevant_large_chunks = []
  for small_chunk_id in initial_results:
    corresponding_large_chunk = get_parent_chunk(small_chunk_id, large_chunks) #find associated large chunk
    relevant_large_chunks.append(corresponding_large_chunk)
  return relevant_large_chunks
```


this pseudocode outlines the general concept of first retrieving smaller, more focused pieces of information and then using that information to find larger, more contextual chunks to provide the LLM with the most relevant information for synthesis


jerry also briefly touched on more advanced stuff like agents and fine-tuning these are things to explore later once you’ve mastered the basics   he stressed the importance of starting with simple fixes like chunk size and metadata before diving into the more complex methods

the overall takeaway was pretty clear  building production-ready rag apps is hard but not impossible   it's about understanding the entire pipeline from data ingestion to synthesis and constantly evaluating your system to improve it also it's all about iteration starting simple and scaling up   it’s like building a really good burger  you need great ingredients  the right techniques and a willingness to keep experimenting until you get it perfect  and maybe a bucket hat.
