---
title: "With ActiveAdmin: How to avoid duplicating declaration between index and exports?"
date: "2024-12-14"
id: "with-activeadmin-how-to-avoid-duplicating-declaration-between-index-and-exports"
---

alright, so you're banging your head against the wall with activeadmin and those pesky duplicate declarations between your index page and exports, huh? i've been there. trust me. feels like you're writing the same stuff twice and it's not exactly the 'dry' principle we all try to stick to when coding. i get it.

i remember this one project, back in the early 2010s, it was a disaster. we were building this e-commerce platform, and everything was fine in the beginning. we were using activeadmin because it was so fast to get it up and running and the client was constantly changing the columns they wanted to see, we had a ton of fields in the database, so our index page declarations were monstrous and then... then they wanted exports. of course, they did. we were using csv and later moved to excel exports and the code became a mess. it was just copy pasting of the same columns from the index block to the export block and the code was so long, we were probably going to have to hire a full-time copy-paster to maintain it. changes in the table would result in double work.

it wasn't a sustainable setup, and it got really bad really quickly and at some point i was looking at the computer screen and i realized that i might get carpal tunnel just from that experience alone. not ideal.

so, let's get down to business. the core problem, as you've probably seen, is that activeadmin doesn't inherently provide a way to declare your columns in a single spot and then re-use it both for the index page display and your exports, but do not worry, there are some elegant solutions to keep things neat and avoid redundancy.

the first approach involves extracting the column configuration into a reusable method. it's pretty straightforward and really effective. you’ll define a method inside the activeadmin resource file and then invoke it from the `index` and from the export block that should be `csv` or `xlsx` blocks, or other formats if that's your fancy. this way, you're only defining your columns once and only having one place to modify the columns.

here's an example of that implementation:

```ruby
  ActiveAdmin.register Product do

    def product_columns
      [
        :id,
        :name,
        :price,
        :stock,
        :created_at,
        :updated_at,
      ]
    end


    index do
      selectable_column
      id_column
      product_columns.each { |column| column column }
      actions
    end

    csv do
      column :id
      product_columns.each { |column| column column }
    end

    xlsx do
      column :id
      product_columns.each { |column| column column }
    end
  end
```

in this first example, `product_columns` method returns an array of symbols, that are the actual columns and we use that to render both index and the csv and excel export blocks. this works wonders if you have a list of simple fields. but, sometimes, you have custom formats or associations that require specific processing. what happens now? well, let's go to the second approach.

now let's add some extra processing to the columns, and in general, having more complex logic. we could have something like a formatted price field or showing the name of an associated record. we can modify the same approach as above, by changing the array we are returning to an array of hashes that will contain additional information or processing for each column. we will be defining the column label and the way it will be rendered as an example.

```ruby
  ActiveAdmin.register Product do
    def product_columns_with_format
        [
        {column: :id, label: 'ID' },
        {column: :name, label: 'Product Name'},
        {column: :price, label: 'Price',  format: ->(product) { number_to_currency(product.price) } },
        {column: :stock, label: 'Stock'},
        {column: :created_at, label: 'Creation Date'},
        {column: :updated_at, label: 'Updated Date'},
        ]
    end

    index do
      selectable_column
      id_column
      product_columns_with_format.each do |column_config|
          column column_config[:label] || column_config[:column], &column_config[:format]
      end
      actions
    end

    csv do
      column :id
      product_columns_with_format.each do |column_config|
        column column_config[:label] || column_config[:column], &column_config[:format]
      end
    end

    xlsx do
      column :id
      product_columns_with_format.each do |column_config|
        column column_config[:label] || column_config[:column], &column_config[:format]
      end
    end
  end
```

this example extends the previous approach, by adding a way to have more complex formatting. you can use the `format` key to specify a proc that does the formatting. this keeps your logic centralized and clean. what happens if you need even more complex logic or have specific requirements for different exports, or even index pages?

well, let's move to the third example and probably the most general one, where instead of defining the `column`, you'll define the data that goes inside it. this one is the most flexible one, since it allows more control over the output. the key is to not use the activeadmin `column` method and render the data yourself, using ruby for that:

```ruby
  ActiveAdmin.register Product do

    def product_data_for_export
      [
       { header: 'ID', data: ->(product) { product.id } },
       { header: 'Product Name', data: ->(product) { product.name } },
       { header: 'Price', data: ->(product) { number_to_currency(product.price) } },
       { header: 'Stock', data: ->(product) { product.stock } },
       { header: 'Created At', data: ->(product) { product.created_at.to_formatted_s(:long) } },
       { header: 'Updated At', data: ->(product) { product.updated_at.to_formatted_s(:long) } },
      ]
    end


    index do
      selectable_column
      id_column
      product_data_for_export.each do |column_config|
        column column_config[:header] do |product|
          column_config[:data].call(product)
        end
      end
      actions
    end


    csv do
        column :id
      product_data_for_export.each do |column_config|
         column(column_config[:header]) { |product| column_config[:data].call(product) }
      end
    end

    xlsx do
      column :id
     product_data_for_export.each do |column_config|
        column(column_config[:header]) { |product| column_config[:data].call(product) }
      end
    end
  end
```

in this final example, we're defining `product_data_for_export` which is an array of hashes, each one representing a column with a header and how the data should be obtained for that column. the data is a proc or lambda that you can customize and get access to all the record data. this approach gives you maximum flexibility and allows you to customize the logic for specific exports, if you really need to. with this, we have successfully decoupled the column definition from the rendering of the column and centralized all the business logic. this way you can test and refactor with more ease.

i've used all of these approaches in different projects and i would go with the third approach in general, as it provides maximum flexibility and you only define what data goes in each column and can refactor the rendering in the future if required and you can do some crazy complex calculations if needed.

now, regarding resources, i would recommend a few places to really deepen your knowledge. first, dig into the ruby documentation for metaprogramming, if you plan to use the method-based solution extensively, specially the `proc` documentation and how to use `call`.  also, familiarize yourself with the activeadmin documentation, in particular the section about index pages and exports, even though it does not go deep enough in these subjects. you can also read "eloquent ruby" by russ olsen. i've found that book quite useful when i had to deal with similar situations. it's a classic for a reason. also check out "refactoring: improving the design of existing code" by martin fowler, its fundamental if you plan to refactor your code and keep it clean and easy to understand.

one last thing and i'm done i promise, when i first started coding, i thought debugging was my superpower because i spent more time finding bugs that actually writing code, but then i discovered that actually, writing cleaner code and reducing duplications helps reduce bugs a lot. well, good luck out there and happy coding!
