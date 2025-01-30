---
title: "How can I optimize Neo4j queries with multiple optional matches?"
date: "2025-01-30"
id: "how-can-i-optimize-neo4j-queries-with-multiple"
---
I've spent considerable time wrestling with the performance implications of complex, optional relationships in Neo4j, and I've found that a careful approach to structuring your queries, particularly involving multiple `OPTIONAL MATCH` clauses, is paramount for maintaining speed. The core challenge arises from the potential for Cartesian products when these matches don't find corresponding nodes or relationships, forcing the database to explore many possible paths that don't exist.

The primary optimization strategy revolves around reducing the number of paths that Neo4j has to evaluate. Instead of allowing the query engine to explore all combinations of optional matches, I often find it more performant to pre-filter the dataset, and in some cases perform conditional merging or pattern definition, to ensure we're only focusing on relevant combinations.

The issue isn't inherently with `OPTIONAL MATCH` itself, but rather its usage without proper constraint. A simple example will illustrate: consider a scenario where we have `User` nodes, `Post` nodes, and `Comment` nodes. A user might optionally have authored posts or commented on posts. A naive approach to finding users along with their optional post and comment information may look like this:

```cypher
MATCH (user:User)
OPTIONAL MATCH (user)-[:AUTHOR]->(post:Post)
OPTIONAL MATCH (user)-[:COMMENTED_ON]->(comment:Comment)
RETURN user, collect(post) as posts, collect(comment) as comments;
```

This query, while functional, suffers when a significant number of users have neither posts nor comments. The `OPTIONAL MATCH` clauses still attempt to find matching patterns for each user, leading to unnecessary processing. Let's take a look at some strategies to address this, using different approaches depending on the desired outcome.

**Example 1: Conditional Matching and Collection for Efficient Optional Data Gathering**

The initial issue with the above query is that all potential paths are evaluated even if no relationships exist. We can significantly improve this by filtering users upfront, by combining match conditions, and only retrieving the optional information if a relationship exists.

Here’s the improved approach which uses a single `MATCH` clause to handle several possible patterns, and `CASE` to selectively populate collections.

```cypher
MATCH (user:User)
OPTIONAL MATCH (user)-[author:AUTHOR|COMMENTED_ON]->(node)
WITH user, collect( CASE
    WHEN type(author) = 'AUTHOR' THEN node
    ELSE null
    END) AS posts,
collect(
    CASE
        WHEN type(author) = 'COMMENTED_ON' THEN node
        ELSE null
        END
    ) AS comments
RETURN user, filter(x IN posts WHERE x IS NOT NULL) as posts, filter(y IN comments WHERE y IS NOT NULL) as comments
```

In this enhanced query, a single `OPTIONAL MATCH` handles both the author and commenter relationships, then `CASE` statements are used to categorize the resulting nodes into respective collections. `NULL` values, which appear when no matching pattern is found, are then filtered. By combining both relationships into a single match, we only iterate over the user nodes, regardless of the number of post or comment relations each possesses, or their lack thereof. This reduces the number of potential Cartesian products the query engine must consider. This is a considerable gain for larger datasets.

**Example 2: Using `WITH` to Filter and Constrain Optional Matches**

In scenarios where it's critical to extract only users involved in specific types of relationships, employing a `WITH` clause after the initial match can drastically improve performance. This lets you effectively "prune" unnecessary computation before executing the subsequent `OPTIONAL MATCH` clauses.

Let's imagine a modified version of the example. Suppose you only care about users who have either created a post or left a comment. The following demonstrates how `WITH` can effectively filter this data, and only afterwards proceed with optional matches.

```cypher
MATCH (user:User)
WITH user
WHERE exists((user)-[:AUTHOR]->()) OR exists((user)-[:COMMENTED_ON]->())
OPTIONAL MATCH (user)-[:AUTHOR]->(post:Post)
OPTIONAL MATCH (user)-[:COMMENTED_ON]->(comment:Comment)
RETURN user, collect(post) AS posts, collect(comment) AS comments
```

Here, the first `MATCH` fetches all users, but the `WITH` clause introduces filtering based on the existence of the desired relationships. Only those users who meet the criterion are passed along to the subsequent `OPTIONAL MATCH` clauses. This drastically reduces the work Neo4j has to perform because it doesn't have to attempt optional matching for users without the specified relations. This type of pre-filtering, by effectively reducing the set of starting nodes for optional matches, can significantly improve the speed of the query.

**Example 3: Utilizing Conditional `MERGE` for Selective Relationship Creation**

While most optimization concerns revolve around read performance, it is useful to include how `MERGE` might be used to optimise some write related use cases. In some specific situations, you might need to create relationships conditionally, but want to avoid creating them if the relationship already exists. `MERGE` can be used here, in conjunction with `OPTIONAL MATCH` to perform selective updates, avoiding redundant matching.

Let’s assume we want to create `LIKES` relationships between users and posts, but only if a user has commented on a post. The below query handles this case efficiently:

```cypher
MATCH (user:User)-[:COMMENTED_ON]->(post:Post)
MERGE (user)-[:LIKES]->(post)
RETURN user,post
```
In the above example, we begin by matching all users that have a `COMMENTED_ON` relationship with a `Post`. We then attempt to create a `LIKES` relationship between user and post using `MERGE`. If this relationship does not already exist, then it is created. If the relationship already exists, then the query does not error, instead, the query simply returns the nodes without making any changes.

In this case, using `MERGE` allows for selective updates to the graph without the need to check for the existence of relations explicitly, and helps to avoid the performance penalty associated with a query that must first look for the relationship before writing it.

**Resource Recommendations**

To further improve understanding of Neo4j query optimization, I recommend exploring resources that focus on query planning. Specifically, examining how `PROFILE` and `EXPLAIN` can reveal the execution plan chosen by Neo4j will aid in identifying performance bottlenecks. Resources that focus on indexing strategies and label usage will also help reduce the search space of a query. Moreover, gaining a deeper understanding of Cypher's execution model and the differences between `MATCH` and `OPTIONAL MATCH` behaviors can greatly improve query writing. It is also worth researching Neo4j's documentation surrounding how to use conditional statements and `WITH` statements effectively in conjunction with one another. Finally, performance tuning guides provided by Neo4j documentation often provide practical examples and techniques for optimizing query performance that build on the above mentioned knowledge.
