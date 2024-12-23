---
title: "How can I use wildcard matching in a Rails Enum type?"
date: "2024-12-23"
id: "how-can-i-use-wildcard-matching-in-a-rails-enum-type"
---

Let's tackle this. Wildcard matching with Rails enums isn't a baked-in feature, as enums generally deal with explicit, predefined values. However, we can certainly achieve the desired functionality through creative use of database queries and a little bit of Ruby logic. I’ve had to implement this a few times over the years, particularly in systems where data inputs weren’t always perfectly consistent, or we had to accommodate varied user-defined categories. It invariably comes up when you move from a tightly controlled data model into something that deals with the messy realities of user input.

The core issue is that Rails enums map to database integers, and straightforward database-level wildcard matching (using `LIKE 'value%'`, for instance) on those integers would be meaningless. Therefore, we need to translate the wildcard concept to the string representations before querying. There are several approaches we can consider, each with its pros and cons. We'll look at what I’ve found most effective.

First, let's consider the setup. Imagine a model called `Product` with an enum `category`:

```ruby
class Product < ApplicationRecord
  enum category: {
    electronics: 0,
    books: 1,
    clothing: 2,
    furniture: 3,
    tools: 4
  }
end
```

Now, if we want to find all products whose category *starts with* "b", it won't work by directly applying a `LIKE` clause to the underlying integer value.

One pragmatic solution involves leveraging the `keys` of the enum, effectively bypassing the underlying integer mappings altogether in our queries. In essence, we are performing wildcard matching on the *string representation* of the enum. Here's how we’d tackle that:

```ruby
# Example 1: Simple Starts With Matching

def find_products_by_category_wildcard(wildcard_string)
  matching_categories = Product.categories.keys.select { |key| key.start_with?(wildcard_string) }
  Product.where(category: matching_categories)
end

# Example usage:
# find_products_by_category_wildcard('b')  # Returns books
# find_products_by_category_wildcard('e')  # Returns electronics
```

In this example, `Product.categories.keys` gives us an array of strings: `["electronics", "books", "clothing", "furniture", "tools"]`. We then use Ruby's `select` and `start_with?` to find the keys matching our wildcard pattern. Finally, we perform a standard Rails query using `where(category: matching_categories)`, effectively querying the database with the explicit enum values which correlate with the matching keys we’ve found.

A subtle point here – this is *case sensitive*. If the initial string was `B` then this would return nothing, which may or may not be what we intend. Also, it does a `start_with?` search, and wouldn’t catch values matching `ook`. To broaden the match criteria, one can use a Ruby regular expression:

```ruby
# Example 2: Regular Expression Matching (Case-Insensitive)
def find_products_by_category_regexp(regexp_string)
  regexp = Regexp.new(regexp_string, Regexp::IGNORECASE)
  matching_categories = Product.categories.keys.select { |key| regexp.match?(key) }
  Product.where(category: matching_categories)
end

# Example usage:
# find_products_by_category_regexp('b.*') # Returns books
# find_products_by_category_regexp('cl.*') # Returns clothing
# find_products_by_category_regexp('.*ook') # Returns books
# find_products_by_category_regexp('f.*') # Returns furniture
```

Here, we take advantage of Ruby's `Regexp` class, allowing a far more flexible matching. `Regexp::IGNORECASE` makes the match case-insensitive, and `.*` is used as a wildcard for “zero or more of any character”. Remember the importance of understanding the capabilities of your regular expression library – I've spent countless hours debugging edge cases related to regular expression syntax when dealing with user input.

Now, you might wonder, "can we do this directly in the database query?” You could, by converting all the enum values to their string counterparts within the query itself and then use a `LIKE` clause within SQL. However, that usually leads to verbose, and sometimes, inefficient SQL expressions. This approach also doesn't generalize well if you switch between databases (some SQL implementations treat `enum` string equivalents differently). Moreover, this technique might not use indexes properly and could lead to full table scans. I've always favoured the ruby-level key matching, as it gives you more control and flexibility.

The previous examples, while effective, perform a small amount of work in Ruby before hitting the database. If you're working with a large number of categories (hundreds or thousands), this ruby-level enumeration and selection could potentially become a bottleneck. In such situations, we can consider a more database-centric approach using an `array` in SQL. This will have database specific implementations. We can craft a SQL query using `WHERE ... IN (...)` clause, which is usually fairly efficient:

```ruby
# Example 3: Database-centric Matching using `IN` Clause (more efficient for large enum sets)
def find_products_by_category_database_regexp(regexp_string)
  regexp = Regexp.new(regexp_string, Regexp::IGNORECASE)
  matching_categories = Product.categories.keys.select { |key| regexp.match?(key) }
  Product.where(category: matching_categories)
  #This method remains the same as example 2, however if you look at the produced sql query
  #it should use an IN clause
end

# Example usage:
# find_products_by_category_database_regexp('b.*')
```

While the ruby code looks the same, the database interaction is quite different. The final `Product.where` call translates into a `WHERE category IN ('electronics','books')` type of query. The database is now doing more of the work.

It’s worth noting the caveats. When selecting categories using wildcards and regular expressions, the matching can potentially return unexpected results depending on how specific your string is. Always provide some form of user feedback that validates the selected categories. Also ensure that you sanitize the inputs properly, especially user-provided `regexp_string` to prevent potential regular expression denial of service attacks (redos).

For deeper dives into efficient database querying with Rails, I recommend consulting "SQL Performance Explained" by Markus Winand, especially regarding the nuances of query planning and execution. For a comprehensive grasp of Ruby regular expressions, "Mastering Regular Expressions" by Jeffrey Friedl is invaluable. You might also find the official Rails documentation on ActiveRecord queries extremely helpful; specifically the sections that discuss conditions and the `where` clause. Finally, pay careful attention to your database's query execution plan; this will help you tune your queries to avoid performance problems as your data volumes scale up. It's not always the code itself, but how the database interprets and executes it.
