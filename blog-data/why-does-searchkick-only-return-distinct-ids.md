---
title: "Why does Searchkick only return distinct IDs?"
date: "2024-12-16"
id: "why-does-searchkick-only-return-distinct-ids"
---

Okay, let's talk about Searchkick and why it, by default, prefers to serve up distinct document ids. This is something that's tripped up quite a few folks, myself included back in the early days when I was scaling up a search feature for a platform I was working on; think a large e-commerce site with millions of products. We quickly moved to Elasticsearch for full-text search, and Searchkick became our go-to gem for easier integration with our Rails backend. The initial results looked great, blazing fast queries... until we noticed that some products, which should have appeared multiple times given the query, were only showing up once. Cue the head-scratching.

The core of the issue isn't actually a bug or oversight; it’s a design choice rooted in performance and how Elasticsearch itself structures data and returns results. When you're performing a full-text search across large indexes, Elasticsearch scores each document based on the relevance to the query. Each document, however, is inherently tied to a specific unique id (or _id, as it's often referred to in the Elastic world). Searchkick, by design, leverages this and defaults to returning the most relevant document for a given id, effectively enforcing uniqueness of document ids within the result set. This behavior is intended to prevent duplicate results based on the same document id across potentially multiple fields.

This is not to say that duplicate data is impossible with Elasticsearch - far from it. Consider the scenario where a product might have the same keywords spread across various fields such as its name, description, and category tags. Without this default behavior of distinct ids, it’s entirely possible (and in fact, probable) you’d see multiple entries in the results for the same exact product – just because the query matched differently on its various fields. This would be a poor user experience and lead to unnecessary complexity in handling the results. This decision by Searchkick isn't malicious or lazy. Instead, it optimizes for what is, in most cases, the expected output when you're dealing with structured data: a single, highly relevant entry per unique document.

Now, you might be thinking, "Okay, makes sense, but what if I *do* need the duplicate matches and just want the score?" The good news is that Searchkick, being a thin wrapper, gives you the tools to tweak the underlying query. We can tell it to stop trying to enforce uniqueness and return each and every matching entry, by using what we call "group by" with a field that is always unique.

Let’s get into some examples:

**Example 1: The default, distinct id return**

Let’s say we have a `Product` model indexed by Searchkick, and we have products with similar keywords across multiple fields.

```ruby
class Product < ApplicationRecord
  searchkick
end
```

```ruby
Product.create(name: "Awesome Red Widget", description: "A fantastic red widget for all your needs", category: "widgets", id: 1)
Product.create(name: "Big Red Widget", description: "A larger version of the popular red widget", category: "widgets", id: 2)
Product.create(name: "Super Red Widget", description: "Another red widget, with advanced features", category: "widgets", id: 3)
Product.create(name: "Red Shoe", description: "A comfortable shoe, color red.", category: "shoes", id: 4)
Product.reindex # indexing products after creation
```

If we perform a search for “red widget”, the typical result (using the default distinct id functionality) will only give us one result *per document id* even though multiple fields within those documents might be matching the criteria.

```ruby
results = Product.search("red widget")

# results.count might be 3 (matching id 1, 2, and 3)
# results would include the records for product 1, 2, and 3 (but not, say 1 twice, just because its name and description both matched).
```

**Example 2: Explicitly requesting non-distinct results (with `group by: :id`)**

To see the "duplicates" (or, more accurately, multiple results based on different field matches for the *same* id), we need to use the `group_by: :id` option. This explicitly tells Searchkick to not collapse results based on the document id, allowing Elasticsearch to return each document with different match scores based on which field matched the criteria for the query. This forces Elasticsearch to consider each matching result independently, instead of just one result per id. In reality, the results still come from a single document per id, it just includes each score from all fields within a single document.

```ruby
results = Product.search("red widget", group_by: :id)

# Note: 'group_by' will not return multiple documents with the same id but with different results.
#       It will only return all results from a single document with the same id (but different fields).

# results.count might be 3 (matching id 1, 2, and 3)
# results will include each result from documents with id's 1, 2, and 3; each will have a different _score for every field matched.
# you could see more matches if a document matched the criteria multiple times (in different fields); however, this won't be a duplicate id.
```

**Example 3: Illustrating the effect of `group_by: :id` with different query**

It's important to understand, however, that even with `group_by: :id`, a single document can only have a maximum number of results equal to the number of searchable fields. For the example we are dealing with, a single document can match at most, 3 times in the "red widget" scenario, once in the name, once in the description, and once in the category. If a document matches the criteria multiple times, it will have different scores per field matched, but only the same document ID is returned. In most cases, it's more useful to group by the *searchable fields* to prevent duplicates. Let's say that we have different products with similar category that we want to return:

```ruby
Product.create(name: "Blue Shoe", description: "A comfortable shoe, color blue.", category: "shoes", id: 5)
Product.create(name: "Green Shoe", description: "A comfortable shoe, color green.", category: "shoes", id: 6)
Product.create(name: "Awesome Red Widget", description: "A fantastic red widget for all your needs", category: "widgets", id: 1)
Product.create(name: "Big Red Widget", description: "A larger version of the popular red widget", category: "widgets", id: 2)
Product.create(name: "Super Red Widget", description: "Another red widget, with advanced features", category: "widgets", id: 3)
Product.create(name: "Red Shoe", description: "A comfortable shoe, color red.", category: "shoes", id: 4)
Product.reindex
```

```ruby
results = Product.search("shoes", group_by: :category)
# results.count might be 1
# results will return a single record, but with results grouped by its category, it will show a single "shoes" category.

results = Product.search("shoes")
# results.count might be 4
# results will return each of the products with a "shoes" category.
```

The key here is that `group_by` doesn't really return duplicate id's. Instead, it returns all the scores for each field with different scores, and only returns a single id. This is why in most scenarios it's more useful to group by searchable field to retrieve the score per field.

For further learning and deeper understanding of how Elasticsearch works and how Searchkick builds on it, I'd highly recommend delving into the official Elasticsearch documentation itself. Specifically, the sections on “full-text search”, “scoring”, and “aggregations” (which is what `group_by` uses under the hood) are crucial. Additionally, the book "Elasticsearch: The Definitive Guide" by Clinton Gormley and Zachary Tong provides a great, comprehensive overview of Elasticsearch's inner workings and the rationale behind its design.

In closing, the behavior of Searchkick returning distinct IDs isn't a limitation, but rather a deliberate choice aimed at delivering expected and efficient search results in most common scenarios. Knowing how to leverage the `group_by` option (and understanding *why* you might want to do so) will allow you to work with Searchkick more effectively, and create a better user experience for your application. It’s a lesson that cost me a few late nights of debugging early on, but it ultimately became one of the fundamental aspects of our search infrastructure.
