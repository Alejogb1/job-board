---
title: "How to implement a search model in Ruby on Rails using a multiple-word query string?"
date: "2024-12-23"
id: "how-to-implement-a-search-model-in-ruby-on-rails-using-a-multiple-word-query-string"
---

Alright, let’s talk about crafting a robust search model in rails, particularly when dealing with multi-word query strings. This is something I've tackled quite a few times, from the early days of rails 2.x all the way through current iterations. It’s more intricate than just slapping `LIKE %query%` in a sql query, especially when you’re aiming for performance and relevance.

The core challenge lies in effectively translating a user's multi-word query into actionable database searches. A naive approach, like splitting the query by spaces and doing a `OR` operation on every word, usually leads to poor performance and often delivers irrelevant results. A much better approach revolves around understanding your data, indexing it correctly, and crafting targeted queries.

Before we delve into code, let’s establish some foundational points. First, for any reasonably sized application, full-text indexing is paramount. Rails' built-in active record mechanisms will struggle with complex queries at scale. For this reason, PostgreSQL’s `tsvector` and `tsquery` features, or a dedicated search engine like Elasticsearch, become necessary. My experience is heavily skewed towards postgres, so I’ll focus on that, but the principles translate well. Secondly, relevance matters, not just a simple presence of keywords. Consider ranking results based on how closely they match the entire query, not just on whether any of the words were present. Lastly, we should strive for a solution that is not only performant but also maintainable. Avoid building convoluted SQL strings directly in your models. Lean on activerecord’s query interface and consider techniques like scopes to encapsulate search logic.

Let's illustrate this with some hypothetical code. Say we're managing a collection of `Product` models, each with attributes like `name`, `description`, and `category`.

**Example 1: Basic Full-Text Search Implementation (PostgreSQL)**

First, add a text search index to your `products` table using a migration:

```ruby
class AddTextSearchIndexToProducts < ActiveRecord::Migration[7.0]
  def change
    add_column :products, :search_vector, :tsvector
    add_index :products, :search_vector, using: :gin

    execute <<-SQL
      CREATE TRIGGER products_search_vector_update
      BEFORE INSERT OR UPDATE
      ON products
      FOR EACH ROW
      EXECUTE FUNCTION tsvector_update_trigger(search_vector, 'pg_catalog.english', name, description, category);
    SQL
  end
end
```

This creates a `search_vector` column, an inverted index for quick searches, and a trigger to update it automatically when products are created or modified. Now, in your `product.rb` model:

```ruby
class Product < ApplicationRecord
  scope :search, ->(query) {
      return all if query.blank?

      sql_query = <<-SQL
      products.search_vector @@ to_tsquery('english', :query)
      SQL

      where(sql_query, query: query)
  }
end
```

This defines a scope called `search`. It uses the `to_tsquery` function (which handles multiple words intelligently) and matches it against the `search_vector`. This avoids the pitfalls of simple LIKE clauses.

You'd use it like this: `Product.search("red shoe").all`

**Example 2: Advanced Full-Text Search with Ranking**

Let’s go a step further and incorporate relevance ranking. This usually means using `ts_rank` within PostgreSQL. Modify your `search` scope as follows:

```ruby
class Product < ApplicationRecord
   scope :search, ->(query) {
        return all if query.blank?

      sql_query = <<-SQL
        SELECT *, ts_rank(search_vector, to_tsquery('english', :query)) AS rank
        FROM products
        WHERE search_vector @@ to_tsquery('english', :query)
        ORDER BY rank DESC
      SQL

      find_by_sql([sql_query, { query: query }])
    }
end
```

Now, each matching `product` will have a `rank` attribute indicating the relative match score, and the results will be sorted by this rank. Using `find_by_sql` in this specific case avoids activerecord’s interpretation of the generated sql, allowing us to use `ts_rank`. You will need to explicitly load the models using `find_by_sql`.

You'd use it the same way: `Product.search("running shoes").all`

**Example 3: Handling Search Prefixes and Partial Matches**

Sometimes, users might start typing and expect results to appear instantly. For this, we would leverage PostgreSQL’s prefix matching using `plainto_tsquery` with some modifications, for example to match a `red` before `red sneakers`:

```ruby
class Product < ApplicationRecord
  scope :search, ->(query) {
    return all if query.blank?

    sql_query = <<-SQL
      products.search_vector @@ plainto_tsquery('english', :query || ':*')
    SQL
    where(sql_query, query: query)
  }
end
```
Here `plainto_tsquery` generates a query that is more suitable for simple input, and the `:*` at the end allows it to match prefixes.

You'd use it just as before: `Product.search("sne").all`

These examples are by no means the end-all, be-all of search implementations. There are many other optimizations one might consider. For instance, stop word removal, stemming, and using custom dictionaries can all increase the effectiveness of your searches. For larger systems, I would strongly suggest reading up on techniques employed in systems such as Elasticsearch, which handles large-scale search applications incredibly well, and could be more suitable for complicated scenarios. Remember though, even if you use elasticsearch, you'll need to be very careful on how you form your search queries - so the general principles above still apply.

For deeper understanding on full-text search in postgresql, I highly recommend the [official PostgreSQL documentation](https://www.postgresql.org/docs/current/textsearch.html). It is, without a doubt, the most authoritative resource. Additionally, consider reading "Relational Theory for Computer Professionals" by C.J. Date for a better conceptual understanding of database concepts, which will greatly help with advanced querying techniques. For a good practical guide on indexing and querying data with both SQL and NoSQL databases, "Database Internals" by Alex Petrov is an excellent resource.

In summary, building a robust multi-word search in rails involves more than just simple string matching. Leveraging full-text indexing, understanding the intricacies of your query engine, and implementing strategies to improve both performance and relevance are essential steps. I hope this helps you tackle your own search challenges.
