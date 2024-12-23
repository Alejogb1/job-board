---
title: "Why does Searchkick only return distinct ID values?"
date: "2024-12-23"
id: "why-does-searchkick-only-return-distinct-id-values"
---

, let’s unpack this. It’s a question I've actually dealt with quite a bit in the past, especially when working on systems that required complex data relationships alongside full-text search. It’s a common gotcha when you start pushing Searchkick’s default behavior. The core of the issue revolves around how Searchkick, at its heart, interacts with Elasticsearch, and why, for performance and conceptual clarity, it typically only returns distinct id values by default.

Searchkick is essentially a gem that sits as a convenient abstraction layer on top of Elasticsearch, and it makes indexing and querying data relatively straightforward. However, Elasticsearch is, underneath, designed to handle massive amounts of data, and its architecture is deeply optimized for its inverted index structure. This structure isn't naturally set up to return multiple identical identifiers. When Searchkick performs a query, it's ultimately triggering a query against this index structure.

The reason you're seeing only distinct ids returned by Searchkick stems from how Elasticsearch processes search results and aggregates them into what it calls “hits.” When you perform a standard search that might match multiple instances of the same model (for example, if you have a model called 'Product' and multiple products share the same ‘name’ value and that name is being searched), the result set Elasticsearch returns will typically contain multiple document hits that technically represent different instances. However, Searchkick defaults to reducing those hits by their primary identifier. This is a deliberate design choice to ensure that you don't end up with a results array that contains multiple references to the *same* underlying record in your database.

Why do that? Well, think about a large e-commerce platform. Suppose you have numerous products with the same generic name, perhaps variations of "red shirt," each with unique product IDs. If a user searches for “red shirt,” without some processing the Elasticsearch result set might contain *multiple* hits representing each instance of the “red shirt” product. This would mean your application could potentially render duplicate data in your application when displaying search results. Searchkick prevents this by ensuring that each *unique* product id appears once. You wouldn’t want 50 slightly different representations of the same database record in your search result list, as it would be essentially useless and wasteful.

However, this also presents an immediate challenge. Sometimes, you *do* need more than just the distinct identifiers. Let's delve into how to overcome this, and I'll illustrate with a few concrete code examples.

**Example 1: Understanding the default behavior and basic workaround**

The first example will use a simple `Product` model and assume that the `name` and `description` fields are searchable.
```ruby
# Assume a Rails app or a similar Ruby environment

class Product < ApplicationRecord
  searchkick
end

# Example usage:
Product.reindex # For demonstration, ensure search index is up-to-date
Product.create(name: 'Red Shirt', description: 'A classic red shirt', sku: 'RS001')
Product.create(name: 'Red Shirt', description: 'Another red shirt but slightly different', sku: 'RS002')

# Default Searchkick behavior:
results = Product.search("red shirt")
puts "Distinct IDs: #{results.map(&:id)}" # Output: [1, 2] - Assuming these IDs

# Workaround if you just need the records:
all_results = Product.search("red shirt").records
puts "All records: #{all_results.map(&:name)}" # Output: ["Red Shirt", "Red Shirt"]
```

As you can see, `.search()` only produces a distinct id result, but using `.records` will give us the full record set. This is because `search` uses a streamlined approach to give you the necessary pointers to load from the DB efficiently, and loading up whole records when only the id is needed is unnecessary.

**Example 2: Retrieving more information from Elasticsearch**

Sometimes, simply getting the records isn't enough; we might need additional metadata returned by Elasticsearch. The default distinct id behavior is still a problem. Here is how to work with it.

```ruby
class Product < ApplicationRecord
  searchkick
end

# Example Usage
Product.reindex
Product.create(name: 'Blue Shirt', description: 'A simple blue shirt', sku: 'BS001')
Product.create(name: 'Blue Shirt', description: 'A nice blue shirt', sku: 'BS002')
Product.create(name: 'Blue Jeans', description: 'Classic blue jeans', sku: 'BJ001')

results = Product.search("blue", body: {
                            query: {
                                multi_match: {
                                    query: "blue",
                                    fields: ["name^3", "description"]
                                    }
                                 },
                             highlight: {
                                 fields: {
                                   name: {},
                                   description: {}
                                }
                             }
                             })

puts "IDs: #{results.results.hits.hits.map { |hit| hit["_id"]}}"
puts "Highlights: #{results.results.hits.hits.map { |hit| hit["highlight"]}}"

```
This code snippet demonstrates using the low-level `body` option to pass custom Elasticsearch queries. Crucially, it directly accesses the Elasticsearch results through `results.results.hits.hits`. This gives you access to the full document data, including highlighting and document scores which Searchkick otherwise filters out. Here, I use the `_id` key to get the original record ids from Elasticsearch.

**Example 3: Advanced scenarios using aggregations (and why the default exists)**

Let's say you want to find the most frequently occurring names. This is where the distinct id filtering *really* becomes important, and where a deeper understanding of Elasticsearch aggregations is required. If Searchkick didn’t return distinct ids, implementing this aggregation would be much harder.
```ruby
class Product < ApplicationRecord
  searchkick
end

# Example Usage:
Product.reindex
Product.create(name: 'Green Shirt', description: 'A green shirt', sku: 'GS001', category: 'apparel')
Product.create(name: 'Green Shirt', description: 'Another green shirt', sku: 'GS002', category: 'apparel')
Product.create(name: 'Green Pants', description: 'A pair of green pants', sku: 'GP001', category: 'apparel')
Product.create(name: 'Blue Hat', description: 'A nice blue hat', sku: 'BH001', category: 'accessories')
Product.create(name: 'Green Socks', description: 'Comfortable green socks', sku: 'GS003', category: 'accessories')

results = Product.search("green", body: {
                      aggs: {
                          product_names: {
                              terms: {
                                  field: "name"
                                  }
                               }
                           }
                      })

results.results.aggregations["product_names"]["buckets"].each do |bucket|
    puts "Name: #{bucket["key"]}, Count: #{bucket["doc_count"]}"
end
```
This demonstrates the use of aggregations, here, calculating how many times each `name` appears in search results for the term "green". While this example doesn't directly deal with the distinct id return issue, it does highlight *why* Searchkick's default behavior is so important for using Elasticsearch effectively. This aggregation is based on the frequency of tokens in the indexed documents and does not care about record ids. Without the distinct id filtering, any further manipulation of record level objects would require extra work on your part.

**Key Takeaways and Further Reading**

The default distinct id behavior of Searchkick exists to optimize for common use cases: avoiding duplicate records, and simplifying the initial result handling. It encourages you to use the `.records` method when you need full models or use the `body` option when more control is needed with Elasticsearch directly.

To really understand this in-depth, I’d recommend diving deeper into:

1.  **“Elasticsearch: The Definitive Guide” by Clinton Gormley and Zachary Tong:** This book provides a thorough understanding of how Elasticsearch works internally, including its indexing structures, query language, and aggregation capabilities. It’s essential for anyone doing any sort of serious work with Elasticsearch.
2.  **The Elasticsearch documentation itself:** The official documentation is incredibly detailed and a key reference point. It includes comprehensive information on every query type, analysis method, and API endpoint. Familiarity with the official docs will quickly reveal how Searchkick translates into the actual queries and results Elasticsearch manages.
3.  **“Taming Text” by Grant Ingersoll, Thomas Morton, and Drew Farris**: This book provides a very in-depth look at information retrieval principles and text analysis with search engines, which is helpful for understanding how Searchkick and Elasticsearch's text processing works under the hood.

By carefully navigating Searchkick’s abstractions while understanding what's going on under the hood in Elasticsearch, you can efficiently build complex search functionality. The default behavior isn’t a limitation; it’s an architectural decision made to streamline your day-to-day interactions with a powerful yet potentially complex technology.
