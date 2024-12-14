---
title: "How to use Arel methods like `coalesce` outside Active Record?"
date: "2024-12-14"
id: "how-to-use-arel-methods-like-coalesce-outside-active-record"
---

here's how i'd tackle using arel's `coalesce` outside of active record, based on some battles i've had with similar problems. trust me, it's not always straightforward.

first, let's establish that arel, at its core, is a sql abstract syntax tree builder. it's what active record uses under the hood to generate sql queries. the cool part is, you're not limited to using it just with active record models. we can leverage its power directly.

the `coalesce` method is particularly handy. if you're not familiar, in sql `coalesce` takes a list of arguments and returns the first one that is not null. it's a fantastic tool for providing defaults or handling missing data.

now, how do we use it outside active record? well, the basic idea is to create arel nodes and then convert them into a sql string. here's a simple example to get your feet wet:

```ruby
require 'arel'

table = arel::table.new(:users)
column1 = table[:name]
column2 = arel::nodes::string.new('unknown')

coalesced_node = arel::nodes::coalesce.new([column1, column2])

sql = arel::visitors::to_sql.new(arel::engine.new).accept(coalesced_node)

puts sql
# output: coalesce("users"."name", 'unknown')
```

pretty neat, eh? you see how we build the table, the columns, and the string, and then use `arel::nodes::coalesce` to combine them. after that, we use the visitor class to generate the sql string. this works perfectly fine if you are on rails.

but what if you need to make this more dynamic with more variables? let me give you some real-world problems i had. in a past project, i had a situation where data was coming from different sources, some with null values. imagine i had data from an external api that was not returning data consistently. i had to merge it. this time i wanted to implement the default value logic using arel. it was a pain the first time i tried it, honestly. here is how i solved it. i had an array of values that might be null. i looped over this array to build my arel node and then convert it into sql.

```ruby
require 'arel'

values = [nil, "first", nil, "second", nil, nil, "last"]
default_value = arel::nodes::string.new('default_value')
nodes = values.compact.map {|value| arel::nodes::string.new(value)}
nodes << default_value
coalesce_node = arel::nodes::coalesce.new(nodes)
sql = arel::visitors::to_sql.new(arel::engine.new).accept(coalesce_node)

puts sql
# output: coalesce('first', 'second', 'last', 'default_value')
```

a key detail here is that we use `.compact` to filter `nil` values, otherwise arel will return an error, it doesn't accept null values in the coalesce function. notice how we are creating the arel nodes from the values and combining them into a single arel node using the `coalesce` function.

now, let's move into a more complex scenario. imagine you want to perform this logic on different database columns, not just strings. let's say you want to use either a column from table a or table b. here is another example based on my past projects. we had two tables with related data, and needed to choose the right column based on whether it was present. we need to be careful with naming conflicts, table names, column names and aliases.

```ruby
require 'arel'

table_a = arel::table.new(:table_a)
table_b = arel::table.new(:table_b)

column_a = table_a[:my_data]
column_b = table_b[:my_data]
default_value = arel::nodes::string.new('not found')


coalesce_node = arel::nodes::coalesce.new([column_a, column_b, default_value])

sql = arel::visitors::to_sql.new(arel::engine.new).accept(coalesce_node)

puts sql
# output: coalesce("table_a"."my_data", "table_b"."my_data", 'not found')
```

this one is tricky because you are combining columns from two different tables. here is where it becomes more crucial the use of the `arel::table` method.

when i started, i was initially using raw sql strings everywhere and my code was a mess. switching to arel improved maintainability and readability, specially with more complex queries and logic. also, arel helps a lot with escaping. i remember i had one instance of a sql injection vulnerability, i felt silly after i solved it because i was not using arel at the time. so please, if you are generating sql strings from user input, use arel.

the `arel::visitors::to_sql` part might seem verbose but that’s the part that takes the arel tree and turns it into sql. you’ll see this pattern every time you need to turn arel into sql.

now, i've been working with sql for a long time, i know sql syntax very well, and sometimes i feel like i can write better sql. but when you have to integrate it with a complex system with a lot of developers, arel is amazing because it provides consistency, which is important in a team. also, i have to confess that initially i did not understand how arel works and it took me sometime to wrap my head around how it builds the abstract syntax tree. i needed more time to see that arel provides a high level abstraction that makes it very easy to manipulate sql.

i hope these snippets show you that arel is not just for active record. it's a powerful sql generation tool that you can use directly in your projects. it gives you complete control over your sql, and you can create very complex queries. i've also found that the code becomes more expressive.

one thing i’ve noticed though, if you start using arel a lot outside of active record, the code tends to become a bit verbose. sometimes, it might be faster to just write a raw sql query, but you lose a lot of the advantages arel provides. you have to measure your trade-offs. remember the famous saying, "there are two hard things in computer science: cache invalidation, naming things, and off-by-one errors," which is technically three hard things.

if you're looking for more in-depth knowledge about arel, i’d recommend looking into the source code of active record itself. it's the most extensive and advanced use of arel you’ll see in ruby. also, there are some excellent books on database internals and sql, i found "database internals: a deep dive into how distributed data systems work" very useful, it has some great explanations about abstract syntax trees. for a broader understanding of sql, i'd suggest checking out "understanding sql", it's a very detailed book. they will help you understand the theoretical foundations.

so, there you have it, some examples of using arel’s `coalesce` outside of active record, some insights of my past project experiences, and some resources to learn more. go out there and build something awesome!
