---
title: "Is using `#find_each` and `#each` in Ruby on Rails a performance concern?"
date: "2024-12-23"
id: "is-using-findeach-and-each-in-ruby-on-rails-a-performance-concern"
---

Alright, let's tackle this. I've seen this question crop up time and time again in projects, and it's a good one, because the nuances between `#find_each` and `#each` on ActiveRecord relations in Rails can significantly impact your application's performance, especially when dealing with larger datasets. It’s not a simple “this one is always better than that one” situation; context is key.

Let's start by dissecting what each method actually *does*. When you call `.each` directly on an ActiveRecord relation, you’re typically loading *all* the records from the database into memory at once. Think of it as a single, large query that grabs every record matching the conditions of your relation and then iterates through the collection in memory. Now, this works perfectly fine for smaller datasets; you’re unlikely to even notice any performance lag. However, the problems arise when you’re working with tables containing thousands, tens of thousands, or even millions of rows. Suddenly, you’re holding a huge chunk of data in memory, and that can quickly lead to memory exhaustion, slow response times, and an overall sluggish application.

On the other hand, `#find_each` is designed specifically to avoid this issue. It fetches records in batches, typically in chunks of 1000, by default, using a `LIMIT` clause in SQL. This means that instead of loading all the data into memory simultaneously, it retrieves and processes records in smaller, manageable groups. It iterates over each record within a batch, yields to the code in your block, and then proceeds to the next batch. This significantly reduces memory consumption and prevents your application from choking on large datasets.

I recall a rather nasty incident on a previous project involving a user analytics feature. We were using `.each` to iterate over user event logs to generate reports. At first, it worked swimmingly. Our user base was small, and the event logs were manageable. As our user count grew, our server started struggling; requests became sluggish and some even timed out. Profiling quickly pointed to excessive memory usage. Replacing `.each` with `#find_each` for iterating over those event logs resolved the issue immediately; memory usage normalized, and response times dropped drastically. That firsthand experience cemented in my mind the importance of understanding the differences.

To illustrate this further, consider the following scenarios using working code snippets. Let's assume we have a `Product` model with a substantial number of records.

**Scenario 1: Incorrect Usage with `.each`**

```ruby
# Assume we have 10,000 products.
products = Product.all

products.each do |product|
  # Do some processing, e.g., update timestamps.
  product.update(updated_at: Time.current)
end
```
In this first example, `Product.all` loads *all 10,000* product records into memory *at once*. Then, we iterate through them. The problem isn't that the update operation itself is slow, it's the loading of all those records before we even start processing. The server has to hold onto all that data.

**Scenario 2: Correct Usage with `#find_each`**

```ruby
Product.find_each do |product|
  # Do the same processing.
  product.update(updated_at: Time.current)
end
```

Here, `#find_each` fetches products in batches. The default batch size is 1000, so it'll make 10 separate queries. Each query fetches a manageable chunk, processes it, and then moves to the next. This drastically reduces memory footprint and the likelihood of out-of-memory errors, especially when dealing with far larger tables than this example represents.

**Scenario 3: Custom Batch Size with `#find_each`**

```ruby
Product.find_each(batch_size: 500) do |product|
  # Do the same processing, but in smaller batches.
  product.update(updated_at: Time.current)
end
```

This snippet shows that `#find_each` also provides control over the batch size. Adjusting the batch size can be useful for tuning your application's performance under certain workloads. Sometimes a slightly larger or smaller batch may be optimal depending on various factors like network latency to your database server and the type of database you're using.

Now, one caveat is that `#find_each` does have some limitations. It requires that the table has a primary key that is orderable, typically an integer `id` column. If you are dealing with a model without such a key or without a consistent ordering, `#find_each` can be problematic and needs careful consideration or you should utilize `find_in_batches` instead.

To deepen your understanding further, I strongly recommend looking into the following resources:

* **"Rails AntiPatterns: Best Practices and Pitfalls" by Chad Pytel and Tammer Saleh**. This book offers very practical, real-world advice on writing efficient Rails code, including detailed information on optimizing database interactions, exactly the topic that we are talking about.

* **"The Ruby Programming Language" by David Flanagan and Yukihiro Matsumoto**. Having a robust grasp of the underlying principles of the language, especially regarding Enumerable methods, is fundamental. This book provides exactly that foundational information.

* **The official Ruby on Rails guides, especially the Active Record Querying section**. This is your go-to resource for understanding the specifics of ActiveRecord and how to use methods like `#each`, `#find_each` and other tools effectively.

In short, the core distinction is about *how* the data is loaded from the database. If you're dealing with smaller sets of records, `.each` isn't *inherently* problematic. However, as your datasets scale, you absolutely *need* to consider the performance impact. `#find_each` is specifically designed to address the issues of iterating through larger datasets by processing records in batches, preventing your application from getting bogged down. Ignoring this distinction will eventually lead to significant performance issues, which could easily be avoided by making the more efficient choice. It’s a simple change with potentially enormous gains.
