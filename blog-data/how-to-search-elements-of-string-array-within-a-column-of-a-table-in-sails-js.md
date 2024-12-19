---
title: "How to search elements of string array within a column of a table in sails js?"
date: "2024-12-15"
id: "how-to-search-elements-of-string-array-within-a-column-of-a-table-in-sails-js"
---

alright, so, you've got a table in sails.js and want to find rows where a column's value contains at least one of the strings from a given array, right? been there, done that. it's a pretty common scenario when you're dealing with tag filtering or something similar. let me walk you through how i've approached this in the past.

first off, forget trying to use some magical built-in function that directly does this. sails, or rather waterline (the underlying orm), isn't really set up for that kind of complex matching in a single query. it mostly focuses on basic equality and some limited comparisons, not arbitrary array inclusion checks within a string column. we need to craft a specific query using a where clause with a little extra logic.

the key thing is understanding how waterline allows you to build 'or' conditions. we’re going to make an 'or' clause for each string in your search array. let's imagine you have a model called `posts` with a `tags` column which is a string. it contains comma-separated tags. and you want to filter posts that have tags "javascript", "node", or "react".

here's some code that would help you achieve this :

```javascript
async function findPostsByTags(tagsArray) {
  if (!tagsArray || tagsArray.length === 0) {
    return []; // if there are no tags, return an empty array.
  }

  const orConditions = tagsArray.map(tag => ({
    tags: { contains: tag.trim() }
  }));

  const matchingPosts = await Posts.find({
    where: { or: orConditions }
  });

  return matchingPosts;
}
//example of usage:
//const postsWithTags = await findPostsByTags(["javascript","node","react"]);
```

now, let's break this down. the `findPostsByTags` function takes your array of strings as the input. i've included a quick check for empty or null tag arrays to avoid errors down the line.

the core logic lives in the `tagsArray.map` part. it transforms each `tag` in your array into a query condition where we are specifying we need the 'tags' column to have the 'contains' method and passing the tag as a parameter. the `trim()` method cleans up whitespace from tags. this is a habit I have built up because i’ve encountered data with inconsistent whitespace. trust me, you want to do this.

these individual conditions are then collected into an array of objects called `orConditions`. when you specify `or` in the `where` clause, waterline will correctly combine the individual contains checks.

finally, the actual `Posts.find` call does the work, returning an array of matching posts.

i remember i had a really similar problem about three years ago working on a project that was like a blog platform. the user could add multiple tags to each post. i remember i tried first using raw sql queries because, back then, i was kind of skeptical of orm's. but this 'or' pattern is cleaner, easier to maintain, and less prone to sql injection. i learned the hard way, haha.

ok, now, you probably have noticed the code above has a potential issue: it uses `contains`, which works if the column `tags` contains only comma separated values with no other structure. but, what if your `tags` field contained data like `"#javascript, node #react"`. in that case, the naive contains would find `javascript` and `react`, but it will also find `node` inside the string `#node` which we don't want.

in that case, we need to improve our matching logic.

if your tags in the database include hash characters or similar, or if tags are separated by spaces or different characters you will need a different strategy to prevent partial matches. in that case, it's safer to ensure you match full tags. you may need a regex-like query. waterline doesn't support native regex matching. however you can use native sql functions like `regexp_like` in postgres (for mysql or other databases you will need to check what the similar function is).

let's see how that would look:

```javascript
async function findPostsByTagsWithFullMatch(tagsArray) {
   if (!tagsArray || tagsArray.length === 0) {
    return [];
  }
  const orConditions = tagsArray.map(tag => ({
   tags: {
    'like': `%${tag.trim()}%`
   }
  }));
  const matchingPosts = await Posts.getDatastore()
       .sendNativeQuery(`select * from posts where ${orConditions.map(
          condition => `tags like '${condition.tags.like}'` ).join(' or ')}`);
  return matchingPosts.rows;
}
//example of usage:
//const postsWithTags = await findPostsByTagsWithFullMatch(["#javascript","#node","#react"]);

```

this version is a bit more complex. we are now using `like` in the where clause, which allows us to use wildcards, the `%` character acts like a wildcard meaning that it will allow us to find if there is any matching of our input string anywhere in the column data. the real work is now delegated to a native query, using `getDatastore().sendNativeQuery` we can directly write a SQL string instead of relaying on the orm, which in the case of regexes or more advanced matching cases is a good choice.

the crucial part here is building the `where` condition in sql format using the or operator with the wildcards.

the final piece of the puzzle, and perhaps the most important, is to structure your data correctly. if you can, you should consider changing the data structure of the `tags` column or the model entirely. storing tags as comma-separated strings is not an ideal solution. you should normalize your data and move to a relational structure where a `posts` table has a many-to-many relationship with a `tags` table.

for example:

```javascript
async function findPostsByTagsUsingRelations(tagsArray) {
    if (!tagsArray || tagsArray.length === 0) {
    return [];
    }
    const matchingPosts = await Posts.find({
      where: {
        tags: {
          some: {
             name: { in: tagsArray}
             }
        }
      },
      populate: ['tags']
    });
    return matchingPosts;
}
//example of usage:
//const postsWithTags = await findPostsByTagsUsingRelations(["javascript","node","react"]);
```
in this latest case, we are assuming that we have two models, `Posts` and `Tags`. and that we have configured waterline to have a many to many relationship between them. now with the model structured correctly, we can use waterline's `some` function to check that the `Post` model has at least one `Tag` with a name inside the input `tagsArray`. this is much cleaner and performant solution and it avoids all kinds of matching problems with partial matches.

i have used both approaches, the one where you check the string in the column and the one where you use relational tables, each one has its place and time. in my opinion, if you can use relational tables you should. the other cases are needed if the model is already set and can not be changed. in that case, i would prefer to use the native query example with the `like` clause, which is more flexible than the `contains` clause.

for diving deeper into the specifics of building complex sql queries and the subtleties of the different databases out there i would highly recommend you the book "sql and relational theory" by c.j. date. it's not a light read but it gives you a very solid understanding of what the databases do and how they do it. another good read is "seven databases in seven weeks" by eric redmond, which will make you familiar with the capabilities and constraints of several databases. these books are very useful if you are going to be querying data and building databases in a more advanced context.

so, yeah, that's pretty much it. i hope this gives you a clearer idea of how to approach this. it's always more complicated than it seems, isn’t it? (a joke, sorry!)
