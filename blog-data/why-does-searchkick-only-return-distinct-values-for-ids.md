---
title: "Why does Searchkick only return distinct values for ids?"
date: "2024-12-16"
id: "why-does-searchkick-only-return-distinct-values-for-ids"
---

Okay, let's tackle this one. It's a question I've seen come up more times than I can count, and honestly, I can vividly remember the first time I encountered this behavior myself, back when I was trying to build a product catalog search for an e-commerce site. It threw me for a loop. Essentially, you’re observing that Searchkick, by default, returns only unique document ids even when the underlying Elasticsearch index contains multiple documents with the same content but different identifiers. That's the core of the issue. The ‘why’ here isn't some esoteric quirk, but rather a design decision deeply rooted in how Elasticsearch and Searchkick are intended to work, specifically around deduplication and relevance.

The crux of the matter lies in understanding that Elasticsearch, as a search engine, prioritizes the *relevance* of documents based on the search query, not simply retrieving every single document that *matches*. Searchkick, as a higher-level abstraction on top of Elasticsearch, inherits and builds upon this core functionality. When you perform a standard search using Searchkick, it internally issues a query to Elasticsearch. Elasticsearch, in turn, returns search results, which include a score (relevance), id and the source document, among other details. For a typical search use-case, where documents are likely to be unique and represent distinct entities, this is exactly what you'd expect. It’s optimized for typical database-driven applications where each record has a dedicated representation.

However, when documents share the same content but are intended to represent different instances, this default behavior becomes a problem. Suppose you have multiple blog entries with identical content (for, say, a placeholder), and you indexed these with different ids. When searching for a term that appears in all of them, Searchkick will usually only return *one* of those ids, even though all the documents contain the searched term. This stems from Elasticsearch’s default scoring mechanism, which aims to rank more relevant documents higher, combined with Searchkick’s default distinct ids functionality. It doesn't try to return *all* matches of the query, but rather return those that have the best relevance and in the process, de-duplicates results based on their ids.

This behavior, to reiterate, is not an error, but rather a conscious choice optimized for performance and typical use cases where duplicate content with varying ids is uncommon. The logic is: if two documents have identical content, then the one with a higher score (if any) should be considered more relevant, and thus the unique result, or the first of the sorted results will be returned when limiting the result set to one document.

To effectively address this, you can modify Searchkick's behavior and instruct it to return all matching documents by disabling this deduplication. This is commonly done by interacting directly with Elasticsearch parameters within Searchkick. Searchkick provides a flexible interface to do this through the `search` method's options hash, giving you control over the underlying Elasticsearch query.

Here's how to do it in practice, using three code snippet examples that demonstrate common scenarios:

**Example 1: Basic Search Returning Distinct IDs (Default Behavior)**

```ruby
  class Article < ApplicationRecord
    searchkick
  end

  Article.create(title: "Example Article", content: "This is a test content.", id: 1)
  Article.create(title: "Example Article", content: "This is a test content.", id: 2)
  Article.create(title: "Different Article", content: "This content is unique.", id: 3)

  results = Article.search "test"

  puts results.map(&:id) # Output: [1] – Notice it only returns the first result
```

This snippet demonstrates the standard behavior of Searchkick, retrieving just the record with id `1` even though two articles match the `test` keyword.

**Example 2: Returning All Results With The Same Content (Using Option `distinct: false`)**

```ruby
  class Article < ApplicationRecord
    searchkick
  end

  Article.create(title: "Example Article", content: "This is a test content.", id: 1)
  Article.create(title: "Example Article", content: "This is a test content.", id: 2)
  Article.create(title: "Different Article", content: "This content is unique.", id: 3)

  results = Article.search "test", distinct: false

  puts results.map(&:id) # Output: [1, 2] - It returns *both* matching ids.
```

In this example, we explicitly pass the option `distinct: false` to the `search` method. This instructs Searchkick not to perform the default deduplication, hence returning all documents that match the query, even if they share similar content.

**Example 3: Limiting Results While Ensuring Duplicates (Using `per_page` and `limit`)**

```ruby
    class Article < ApplicationRecord
      searchkick
    end

    Article.create(title: "Example Article", content: "This is a test content.", id: 1)
    Article.create(title: "Example Article", content: "This is a test content.", id: 2)
    Article.create(title: "Example Article", content: "This is a test content.", id: 3)
    Article.create(title: "Different Article", content: "This content is unique.", id: 4)

    results = Article.search "test", distinct: false, per_page: 2

    puts results.map(&:id) # Output: [1, 2] - It returns the first two matching ids.

    results = Article.search "test", distinct: false, limit: 2

    puts results.map(&:id) # Output: [1, 2] - It returns the first two matching ids.
```

In the third example, we use `per_page` and `limit` parameters in conjunction with `distinct: false`. This shows that the behavior to fetch all documents which match the query and the behaviour to limit search results are two completely separate aspects and can be configured simultaneously. It will return multiple ids, as requested, even when they have the same content, and even when the results are limited to the requested count.

Regarding further study, I'd recommend exploring the Elasticsearch documentation very closely, particularly the sections covering relevance scoring and search query parameters. "Elasticsearch: The Definitive Guide" by Clinton Gormley and Zachary Tong is an excellent starting point; It is a detailed overview of the core Elasticsearch concepts. For a deeper dive into information retrieval and search algorithms, "Introduction to Information Retrieval" by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze, is considered the standard resource.

In my experience, these resources and the practical examples above will significantly deepen your understanding of Searchkick and its underlying behavior. It's crucial to remember that understanding the choices made in building tools like Searchkick will help you adapt them effectively for your specific requirements. While the distinct ID behavior might initially seem puzzling, it's a design decision that makes sense within the larger context of search engine optimization and efficient data retrieval. Once you have grasped these concepts, you will feel much more confident in adjusting the behavior of tools such as Searchkick and build customized search features.
