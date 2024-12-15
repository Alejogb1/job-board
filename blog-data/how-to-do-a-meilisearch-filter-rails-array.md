---
title: "How to do a Meilisearch Filter Rails Array?"
date: "2024-12-15"
id: "how-to-do-a-meilisearch-filter-rails-array"
---

alright, so you're looking to filter meilisearch results based on an array of values from your rails app, huh? been there, done that, got the t-shirt. meilisearch is fantastic, but getting it to play nice with complex filtering scenarios in rails can sometimes feel a bit like trying to herd cats. i’ve certainly spent a few late nights scratching my head over this one myself.

first off, let’s tackle the core issue: meilisearch filters are designed to work with specific fields and values. they aren’t inherently built to handle filtering against a dynamic array. typically, you’d filter by something like `category = 'electronics'` or `price > 100`. when dealing with arrays, we need to be a bit more creative. think of it as crafting the query to ask meilisearch to look for documents where one of the many values in the array matches an indexed field.

the trick is to generate a filtering string that meilisearch understands. the `in` operator here is our best friend. this operator lets us specify a set of values to match against. if any value of the field is present in this given set, the document is a hit.

let's imagine a real-world scenario. i once had to build a product search feature for a web app, and these products had multiple tag associations, stored in an array as `tag_ids`. users needed to filter by those tag ids, and of course, they didn't want just one, they wanted to filter by several tags. so i had an array like `[1, 5, 10]` and had to search documents where the document had one or more tags in this list of tags.

instead of trying to manipulate the array inside meilisearch, we’ll build a filter string outside of meilisearch, send it to meilisearch as part of our query. here’s a basic example:

```ruby
def search_with_tag_ids(tag_ids)
  filter_string = tag_ids.map { |tag_id| "tag_ids = #{tag_id}" }.join(' OR ')
  MeiliSearch::Rails.client.index('products').search(
    '',
    filter: filter_string
  )
end

# usage:
# search_with_tag_ids([1, 5, 10])

```

what we’re doing here is mapping over the array of `tag_ids` and constructing a string that looks like this: `"tag_ids = 1 OR tag_ids = 5 OR tag_ids = 10"`. meilisearch will then process this string and return only the documents which contain at least one of those tags.

but wait, there’s more. things can get a bit complex if you add more constraints. say you want to combine this tag filter with a category filter, this is where parentheses come into play. we should wrap the whole tag filter in parentheses and use `AND` operator. this makes sure that meilisearch will apply the tag and category filter and not get confused.

```ruby
def search_with_tags_and_category(tag_ids, category)
  tag_filter_string = tag_ids.map { |tag_id| "tag_ids = #{tag_id}" }.join(' OR ')
  filter_string = "(#{tag_filter_string}) AND category = '#{category}'"

  MeiliSearch::Rails.client.index('products').search(
    '',
    filter: filter_string
  )
end

# usage:
# search_with_tags_and_category([1, 5, 10], 'electronics')
```

here, we’re building the tag filter string like before, then combining it with the category filter using parentheses and the `and` operator, creating a more sophisticated filter query.

now, let's talk about what we've just done and what could go wrong. one thing we need to consider is that strings concatenation can lead to syntax errors if our input strings contain characters like single or double quotes. we need to take extra care to sanitize the input values we are receiving. i've had my share of database errors that i've chased for hours, that's not the goal here. you can use `sanitize_sql_array` in rails for this. also, if your app involves users creating tags freely, then you have to be extra vigilant because the input can be anything and you can't trust the user's input.

consider the following sanitization strategy which builds the query using placeholders and then replaces them.

```ruby
require 'active_support/core_ext/object/to_query'

def search_with_tags_and_category_safe(tag_ids, category)
  tag_conditions = tag_ids.map.with_index { |_, index| "tag_ids = :tag_id_#{index}" }.join(' OR ')
  query_params = tag_ids.each_with_index.to_h { |tag_id, index| ["tag_id_#{index}".to_sym, tag_id] }
  query_params[:category] = category
  filter_string = "(#{tag_conditions}) AND category = :category"

  MeiliSearch::Rails.client.index('products').search(
    '',
    filter: filter_string,
    filter_params: query_params
  )
end

# usage:
# search_with_tags_and_category_safe([1, 5, 10], 'electronics')
```

in this version, we use placeholders like `:tag_id_0`, `:tag_id_1` and a single `:category`. then we populate those placeholders with our query params. this is a safer and more reliable approach.

another thing to be aware of is the length of the generated string. if you have very large arrays, the resulting filter string can become long, potentially hitting limits on meilisearch or your infrastructure. in that scenario consider indexing the tags as a single string with separator between them and searching against that. this would require extra steps on the creation and updates but should solve the long filter strings.

also, meilisearch provides a built-in mechanism for handling filters, and they also have an api which receives an array of filters directly, this approach makes the process less error prone and more clean. here's the example:

```ruby
def search_with_tags_and_category_with_filter_array(tag_ids, category)
    filters = []
    tag_ids.each { |tag_id| filters.push(["tag_ids = #{tag_id}"]) }
    filters.push("category = #{category}")
    MeiliSearch::Rails.client.index('products').search(
    '',
    filter: filters
  )
end

# usage
# search_with_tags_and_category_with_filter_array([1, 5, 10], 'electronics')
```

also make sure that the field `tag_ids` is properly defined in the settings of your meilisearch index as filterable.

for resources, i’d highly recommend diving deep into the meilisearch documentation, it contains many useful examples and explains filter syntax in detail. also read more about sql injection techniques and security, understanding how sql injection works help to protect your app. there’s a paper titled “sql injection attacks and defense mechanisms”. additionally, “database system concepts” by silberschatz, korth, and sudarshan is a good general database resource and its principles can apply to almost any database and search engine. it’s heavy, but very very useful if you want to understand the under workings of database indexing and filtering.

i remember one time, i forgot to index the field as filterable and i was getting unexpected results, it took me a while to find that one, so double check your settings on meilisearch as well, this is very important. and the most important thing is to have some logging and monitoring of the queries you send to meilisearch. you will thank me later.

that’s basically how you get meilisearch and rails to play nice when you want to filter results based on an array of values. it requires a bit of query string crafting, but it’s very achievable. just remember to sanitize the inputs and keep an eye on those query string lengths.
