---
title: "How to Rails duplicate multiple records at once and changing certain attributes?"
date: "2024-12-15"
id: "how-to-rails-duplicate-multiple-records-at-once-and-changing-certain-attributes"
---

alright, so you're looking to duplicate records in rails, and tweak some fields while you're at it, huh? yeah, been there, done that, got the t-shirt. it's one of those things that seems straightforward at first glance, but then you’re down in the weeds trying to make it efficient, especially when you're dealing with a pile of records. let me share how i've tackled this in the past, and some patterns that've saved my bacon.

first off, the naive approach. iterating and creating new records one by one. it works. but it's slow, and if you have relations, it becomes a real pain point real quick. i remember one time at an old gig, working on an inventory management system. we had this feature where users could 'copy' a product with all its settings, but just tweak the name and sku. doing that record by record? felt like using a typewriter to write a novel. i learned my lesson. that lesson was databases are a lot better at manipulating data than loops are. and for this reason, i almost always start with an sql insert.

the core idea here is to craft an sql insert statement that does the copying for you, and then use a `where` clause to adjust the attributes as needed. it’s way faster because the database does the heavy lifting. not ruby. here's a basic example that i’ve found useful:

```ruby
def duplicate_records(model_class, ids, changes)
  # construct the sql fragment to clone the record, and the changes
  # this function does not work if the changes involve relations
  sql_fragment = changes.map do |key, value|
      "#{key} = '#{value}'"
    end.join(', ')
  
  quoted_table_name = ActiveRecord::Base.connection.quote_table_name(model_class.table_name)
  quoted_ids = ids.map { |id| ActiveRecord::Base.connection.quote(id) }.join(",")

  sql = <<~SQL
    insert into #{quoted_table_name} (#{model_class.column_names.join(", ")})
    select #{model_class.column_names.join(", ")} from #{quoted_table_name}
    where id in (#{quoted_ids})
  SQL
  ActiveRecord::Base.connection.execute(sql)
    
  update_sql = <<~SQL
    update #{quoted_table_name} set #{sql_fragment}
      where id in (select max(id) from #{quoted_table_name} group by created_at having count(*) > 1)
  SQL

  ActiveRecord::Base.connection.execute(update_sql)

end

# example usage:
# suppose you have a `product` table and you want to copy
# products with ids 1 and 2, and rename them and give a new sku.
# we get all these attributes from the model column names.
# Product model class required
# this will duplicate them and add name=copy_name and sku=new_sku
# where the ids are 1 and 2.
duplicate_records(Product, [1,2], {name: 'copy_name', sku: 'new_sku'})
```

let's break this down a little bit. `duplicate_records` takes the model class, a list of ids to duplicate, and a hash of changes. the first sql query builds an insert from a select. basically, “take all the columns from these existing records, and copy them into a new record.” after that, you need to update the newly created records with the modifications requested in the changes argument, i do that with a update query that updates the records having the same `created_at` column in the table because the query above will have copied the `created_at` attribute from the source record, we select the max `id` because the newly created ids will be the maximum. this function does not work if the changes involve relations, for example if you are passing something like `{user_id:10}`.

the reason i structured it like this? because you can add more conditions inside the where clause if you want to be picky on what you duplicate. also this handles cases where your changes are not only simple strings but numerical values too. i use the connection's quote method to avoid simple sql injections, this can be improved further to handle more complex scenarios of course. this is an sql query, not ruby, so it runs inside the database server.

now, that's cool and all for simple changes. but what if you have associated records? what if you need to duplicate all the related records too? this is where things get trickier. you could handle it with joins on the sql insert, but it quickly becomes more complex to maintain. here's where i often lean towards a "clone" method on the model itself, something like this:

```ruby
class Product < ApplicationRecord
  has_many :product_images
  def clone_with_associations(new_name:, new_sku:)
    new_product = self.dup
    new_product.name = new_name
    new_product.sku = new_sku
    new_product.save!

    self.product_images.each do |image|
      new_image = image.dup
      new_image.product_id = new_product.id
      new_image.save!
    end
    new_product
  end
end

# example usage
product = Product.find(1)
new_product = product.clone_with_associations(new_name: "copied product", new_sku:"copied-123")
puts new_product.id
puts new_product.product_images.map(&:id)
```

here, i use `dup` to create an instance with all the same attributes of the old one. then i set the modified attributes. and then i create a new record and save it. then i iterate through the associations and do the same. the associations logic should be adjusted depending on the data model you have. this approach is more ruby-centric than the sql way, but gives you a lot of control on what gets duplicated and what doesn’t, and how. this is a good thing for complex scenarios. the `dup` method is key, it creates a copy with the same attributes, without saving it to the database. this way, you can modify the fields you need before saving the copy.

now, for real complex stuff, i’ve had to use database functions. basically, a function that does all of this inside the db itself. it’s the same principle, but you push the logic down to where the data actually is. and if you have hundreds of fields, then using a database function that uses `information_schema` to dynamically query your table columns to avoid you having to manually specify them, becomes really useful. although, the function below is overly simplified for just simple column copying, it’s more complex to write this code correctly for all use cases, but i will leave you with an example of a general function for copying all records from one table to another:

```sql
-- function for cloning a table records to another table
create or replace function clone_records(source_table text, destination_table text, where_clause text default null, columns text[] default null)
returns void
as $$
declare
	source_columns text;
	target_columns text;
	sql text;
begin
	if columns is null then
		select array_agg(column_name) into source_columns from information_schema.columns
		where table_name = source_table;
    select array_agg(column_name) into target_columns from information_schema.columns
		where table_name = destination_table;
		
    if array_length(source_columns, 1) != array_length(target_columns,1) then
      raise exception 'The number of columns does not match';
    end if;

	else
      source_columns := array_to_string(columns, ',');
      target_columns := array_to_string(columns, ',');
  end if;

  sql := format('insert into %I (%s) select %s from %I', destination_table, target_columns, source_columns, source_table);
  
  if where_clause is not null then
    sql := sql || ' where ' || where_clause;
  end if;

  execute sql;
end;
$$ language plpgsql;

-- example usage:
--  this example requires that both tables have the same number of columns
-- and same columns names.

-- let's suppose we want to copy all records in table1 into table2.
select clone_records('table1', 'table2');

-- let's suppose we want to copy all records in table1 into table2 only where column_1 = 'test';
select clone_records('table1', 'table2', 'column_1 = \'test\'');

-- let's suppose we want to copy only column_1 and column_2 into table 2
select clone_records('table1', 'table2', null, array['column_1','column_2']);
```

this example requires postgresql, but all the main database servers support sql procedures, functions. it’s a bit of a rabbit hole to get into writing complex database functions, but when you're pushing lots of data around, it can be a big gain. this allows you to perform complex data manipulations without transferring all that data to your application. the joke is that, sometimes the best code is no code.

a word about best practices, always think about the indexes in your database and how all of this will affect your database. it’s good to check the explain output of your sql queries to see the database execution plan. the more data you have, the more this will become crucial. also, batch operations are your friend. if you are dealing with many thousands of records, never insert them one by one. use batches, use transactions for integrity. think about the locks you create on the database, the resources you consume, and the memory usage you make.

for learning more, i'd suggest checking out "understanding sql" by martin gruber, and "relational database design clearly explained" by jan l. harrington for the database side of things. for the ruby side, "metaprogramming ruby" by paolo perrotta is a fantastic source to understand the flexibility and power of the language, things like `dup` and `clone` are very well explained.

so yeah, that's how i've handled duplicating records in rails over the years. start with simple sql for straightforward cases, dive into cloning methods when you need control, and consider db functions for heavy lifting. remember to monitor performance, and always be ready to go back and optimize. happy coding.
