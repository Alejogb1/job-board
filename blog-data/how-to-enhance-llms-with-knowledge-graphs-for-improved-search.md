---
title: "How to Enhance LLMs with Knowledge Graphs for Improved Search"
date: "2024-11-16"
id: "how-to-enhance-llms-with-knowledge-graphs-for-improved-search"
---

dude so i watched this crazy talk about knowledge graphs and llms and how they're changing search it was wild  the whole thing was basically this guy's mission statement—he's spent his whole career getting devs to build better apps using data not just as separate points but connected like a huge web and now he's applying that to llms and gen ai

the setup was all about how search evolved  he started with this throwback to altavista remember that ancient search engine?  a visual cue was a screenshot of the altavista page totally vintage  he talked about the "altavista effect"—too many results making it useless  then google came along with pagerank its genius was using graph algorithms like eigenvector centrality at internet scale this was a pivotal moment that made google the beast it is  another visual cue was a google press release screenshot showing their indexing billions of urls


then came the knowledge graph  another visual was a screenshot of a google search result showing the knowledge panel—this is where google stopped just indexing words and started understanding concepts and the relationships between them think nodes and edges  it's a graph database! they were using things not just strings, that's what got them their next phase of dominance  the knowledge graph is basically a massive graph database showing relationships between concepts like moscone center's address, owner, etc. nodes with properties and edges representing relationships, pretty cool right


now the big shift is graph rag  that's retrieval augmented generation where the retrieval part uses a knowledge graph maybe along with vector search  he gave a customer service bot example—imagine a company with support articles for wifi routers the articles are nodes with properties like the article text and relationships to the product, engineer etc.  a customer asks "my wifi lights are flashing yellow"—vector search finds relevant articles (initial nodes) then the graph traversal expands this to find related articles, maybe from the same engineer or the same product family  all that context goes to the llm which spits out an answer much better than just using plain vector search alone

here's where things get coding  first, let's imagine how we'd represent the "apples and oranges are fruits" statement in vector space vs a graph

vector space representation (totally opaque and conceptual, this is simplified):


```python
import numpy as np

apple_vector = np.array([0.8, 0.2, 0.5])  # some arbitrary vector
orange_vector = np.array([0.7, 0.3, 0.6])
fruit_vector = np.array([0.9, 0.1, 0.7])

# similarity calculation (dot product, simplified)
similarity_apple_orange = np.dot(apple_vector, orange_vector)
similarity_apple_fruit = np.dot(apple_vector, fruit_vector)

print(f"apple-orange similarity: {similarity_apple_orange}")
print(f"apple-fruit similarity: {similarity_apple_fruit}")
```

ok now graph representation(much clearer):


```python
# neo4j cypher query example
CREATE (a:Fruit {name: 'apple'})
CREATE (o:Fruit {name: 'orange'})
CREATE (f:Category {name: 'Fruit'})
CREATE (a)-[:IS_A]->(f)
CREATE (o)-[:IS_A]->(f)

# query to find all fruits
MATCH (n:Fruit) RETURN n
```

see the difference? the graph is readable it shows the relationships explicitly the vector thing is just numbers that mean nothing to a human being, until you know what the vector stands for


next, he talked about building these graphs  there are three types of data: structured (databases), unstructured (pdfs, text), and semi-structured (kinda structured with some long text fields) structured data is easiest to put in a graph, unstructured is the hardest   he showed a new tool  "knowledge graph builder"—you drag and drop pdfs, links, etc and it creates a graph  a visual cue was a demo showing this tool extracting info from a pdf and creating nodes and edges  it was pretty slick  


the code to do this would be a beast, involving lots of nlp, maybe something like spaCy for extracting concepts and relationships, then neo4j or similar for database interaction. here's a tiny part, assuming we already have entities extracted

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password")) #replace with your db creds

def create_relationship(tx, source_node, relation_type, target_node):
    tx.run(f"MATCH (s:{source_node}), (t:{target_node}) WHERE s.name = '{source_node}' AND t.name = '{target_node}' CREATE (s)-[:{relation_type}]->(t)")


with driver.session() as session:
    session.write_transaction(create_relationship, "Person","WROTE", "Article")
    #add more relationship code here...
driver.close()
```

the resolution was that graph rag is amazing for building more accurate and explainable ai apps—higher accuracy because it uses contextual info from the graph easier to build once you have the graph in place, and better for things like auditing and governance  it's a powerful combination of graph databases and llms he emphasized that the graph representation isn't a replacement for vector embeddings it's a complement, they work together very nicely.  it's  a whole new era of search and ai apps  pretty mindblowing stuff really
