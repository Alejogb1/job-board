---
title: "How to Accelerate Scientific Discovery with LLMs"
date: "2024-11-16"
id: "how-to-accelerate-scientific-discovery-with-llms"
---

dude so i just watched this insane talk about using llms for scientific discovery and it was *wild*.  the whole premise was basically "can we use these fancy language models to speed up scientific breakthroughs, like, way faster than the usual eight-year slog?"  the speaker, hubert some dude from pharma novartis, was totally chill and super relatable—kinda like the nerdy genius friend who always has the coolest insights.  he wasn't even trying to be overly formal—more like he was just grabbing a coffee and spilling the tea on some seriously complex stuff.


the setup was this huge paradox in biology—way back in the early 90s, scientists were messing around with petunia flowers trying to make them super vibrant.  they overexpressed a gene expecting a brighter color but instead got a total color flip—like, wtf?!  it was a major head-scratcher, and it took eight years and a nobel prize to sort it out.  that's a whole lot of time, especially when you're trying to, say, cure cancer.


hubert's whole point was—couldn't we have gotten there quicker? can we use these new AI tools, specifically large language models or llms and retrieval augmented generation (rag) systems,  to cut through the decades of research and maybe even predict future discoveries?  he mentions this as a key visual cue:  *the color flip of the petunia flowers*.  another visual was this super clean diagram of the rag pipeline which he mentions later in the talk.   it showed how information flows through the system, something i totally geeked out on.  another cue was his casual mention of a nobel prize, dropping it like it was nothing, which kinda made me realize the scope of what he was describing.



one of the main ideas was about this thing called "reasoning" within a rag system.  now, you know what a basic rag is, right? question goes in, model searches a database, spits out an answer.  but hubert was saying that's too simplistic for complex scientific questions.  he was arguing for something more sophisticated—doing some serious pre-retrieval thinking.  think of it like this: before you even start searching the library, you already have a rough idea of which shelves to check first, right? that's the pre-retrieval reasoning. you're not randomly grabbing books.


here’s where it gets techy—he mentioned the need for "reasoning" before the retrieval step in rag.  he illustrated this with the analogy of a graph neural network which is something he works on.  imagine a graph where nodes are concepts and edges are relationships. the idea of reasoning involves finding meaningful paths or patterns within that graph before searching the entire database of scientific papers.  that makes the search much more efficient. it’s not just keyword matching anymore.


he gave the example of "routing" in question answering pipelines which is a technique you often see in neural networks.  this "routing" helps navigate the semantic space of the question before looking for answers. this all happens *before* the LLM even sees the document and attempts to generate a response. this is what he described as “reasoning”.


to give you a taste, here's a python snippet showing how you might represent a simple knowledge graph using a dictionary:

```python
knowledge_graph = {
    "gene_x": {"function": "flower_color", "related_to": ["gene_y", "rna_interference"]},
    "gene_y": {"function": "enzyme_activity", "related_to": ["gene_x", "petunia"]},
    "rna_interference": {"mechanism": "gene_silencing", "related_to": ["gene_x", "color_change"]}
}

# Now you can perform some basic reasoning using this graph
# for example, you can find all genes related to gene_x
related_genes = knowledge_graph["gene_x"]["related_to"]
print(f"Genes related to gene_x: {related_genes}")
```

this isn't a full-blown reasoning engine, but it shows the basic principle: representing relationships and using them to infer new knowledge.  it's the foundation upon which you can build something more sophisticated.  


the other key idea he brought up was about "groundedness"—the importance of the llm understanding the context and the historical timeline of the scientific discovery.  you can't just let it pull random facts from wikipedia because it would cheat; you need to limit the llm's knowledge to only the information available *before* the actual scientific discovery.


another example code snippet: imagine you're filtering a database of scientific papers based on publication date:

```python
import pandas as pd

# Assuming you have a dataframe with paper information
papers = pd.DataFrame({
    "title": ["Paper A", "Paper B", "Paper C"],
    "year": [1990, 1995, 2000]
})

# Filter papers published before 1996 to simulate the knowledge cutoff
papers_before_discovery = papers[papers["year"] < 1996]
print(papers_before_discovery)
```

this is how you create a controlled dataset for your rag to access. it can’t use information released after 1995.


and then there was the code for actually performing the similarity search itself.  this is where things get really interesting. hubert talked about using the variance in embedding distances to identify the most informative clusters of information.


```python
import numpy as np
from sklearn.cluster import KMeans

# Assuming 'embeddings' is a numpy array of embeddings for all documents
kmeans = KMeans(n_clusters=3, random_state=0).fit(embeddings)
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Calculate the variance within each cluster and between clusters.
# use the cluster with the lowest within-cluster variance
# This would indicate the best cluster of documents most relevant to the query.
variances = np.var(embeddings, axis=0)
best_cluster = np.argmin(variances)
print(f"best cluster index: {best_cluster}")
```

this snippet showcases how one could use clustering techniques to intelligently narrow down the set of documents to a smaller, more relevant subset.  this would be performed after the initial reasoning step to improve efficiency further.



the resolution?  well, it wasn't like a neat "and then we cured cancer!" ending. but the experiment showed that by carefully structuring the questions and incorporating pre-retrieval reasoning steps, they could significantly improve the llm's ability to uncover relevant information and even generate plausible hypotheses. the approach wasn't perfect, but it was a huge step toward using llms to assist in scientific discovery.  it basically implied a workflow: carefully crafting prompts, setting knowledge cutoffs, and applying specific clustering techniques to efficiently process large scientific datasets. using the multi-needle approach for more complex questions.  it highlighted the need for more sophisticated rag architectures tailored to specific scientific problems and not only answering simple questions but generating plausible hypothesis based on literature.

it was a super interesting talk. it blew my mind. even though it didn't have a dramatic finale, it was pretty exciting to see how far we've come with llms and the potential to use these to accelerate research.  it left me with more questions than answers, just like hubert promised, but that's the fun part, right?
