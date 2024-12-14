---
title: "How to add a join model while importing records via CSV in Active Record, Rails?"
date: "2024-12-14"
id: "how-to-add-a-join-model-while-importing-records-via-csv-in-active-record-rails"
---

so, you're hitting that classic rails roadblock, importing csv data that needs to create records in multiple tables at once, including those tricky join models. i've been there, trust me. it’s one of those situations where the initial, simpler approach just explodes in complexity when you add that many-to-many relationship into the mix.

let’s break it down, and i’ll share some stuff i learned the hard way so you don't have to. the fundamental challenge here isn’t just creating the records; it’s doing it efficiently, preventing duplicates, and, crucially, making sure your data is consistent across the board. in a past life, around 2016, i was working on a content management system for a small publisher. their content had complex tag structures and authors, and they wanted to bulk import all their old database using csv files. sounded straightforward at first. of course, reality had other ideas.

first things first, let’s talk about the data structure and what it should look like in the csv. imagine you've got `articles`, `tags`, and a join model `article_tags`. your csv will probably need some columns to identify the article, and then, either a single column containing a comma-separated list of tag names, or a bunch of tag columns, each holding one tag name. for example:

```csv
title,author,tags
"a great article","john doe","technology,programming,rails"
"another one","jane doe","ruby,backend"
```

or

```csv
title,author,tag1,tag2,tag3
"a great article","john doe","technology","programming","rails"
"another one","jane doe","ruby","backend",
```
 the comma-separated approach is easier to manage from the csv end in my opinion.

now, about the code. we're going to process this csv row by row. inside our import logic, we’ll have to find or create existing tags, and then we’ll have to associate them to articles via the join table. we use find_or_create_by! to avoid duplicates, which is a lifesaver. this prevents you from having duplicate entries in the tags table, and avoids breaking the db structure.

here’s a basic example, assuming you have `article.rb`, `tag.rb` and `article_tag.rb` models already set up correctly, and also assuming you’re using the comma separated tags column approach:

```ruby
require 'csv'

def import_articles_with_tags(csv_file)
  CSV.foreach(csv_file, headers: true) do |row|
    article = Article.create!(title: row['title'], author: row['author'])

    tag_names = row['tags'].split(',').map(&:strip)
    tag_names.each do |tag_name|
       tag = Tag.find_or_create_by!(name: tag_name)
       article.tags << tag # using the has_many :through association method

    end
    
  end
end
```

this method iterates over every row of the csv, creates an article with the title and author, then takes the tags, splits them by commas, strips any trailing whitespace, and then iterates over each tag, uses `find_or_create_by!` to get a reference to the existing or create it, and finally adds it to the article using the rails helper.

if you are using the approach with multiple tag columns, you would have to check each column for a tag value that isn't blank and add it to the tags list to be processed, which is more of a pain than this solution.

but there is one problem with this code right now: it isn't very efficient, and if you try to process a large file you will be waiting for the process to complete for what will feel like an eternity. the problem is that for each article, you are executing many queries and those queries are executed inside the loop one by one. we can make it more efficient if we do an eager loading to the tags table at once before iterating and creating our articles. this is a fairly easy thing to fix.

```ruby
require 'csv'

def import_articles_with_tags(csv_file)
    tag_names_set = Set.new # use a set so the tags are unique
    CSV.foreach(csv_file, headers: true) do |row|
        tags = row['tags'].split(',').map(&:strip)
        tags.each{|tag_name| tag_names_set.add(tag_name)}
    end

    #now fetch or create all the tags before creating the articles.
    tags_hash = {}
    Tag.where(name: tag_names_set.to_a).each{ |tag| tags_hash[tag.name] = tag}
    tag_names_set.each do |tag_name|
        tags_hash[tag_name] = Tag.create!(name: tag_name) unless tags_hash[tag_name]
    end


    CSV.foreach(csv_file, headers: true) do |row|
        article = Article.create!(title: row['title'], author: row['author'])

        tag_names = row['tags'].split(',').map(&:strip)
        tag_names.each do |tag_name|
            article.tags << tags_hash[tag_name]
        end
    end
end
```

now we make use of a `Set` that prevents repeated tag names to appear and iterate over the csv first to gather all tag names before fetching them all at once from the database using the `where` clause. we are also using a hash that has the name as a key and a tag as a value, so it is easy to create them and get a reference in the second iteration. now the tags are fetched and created all at once, and there's no need to be doing a lot of queries in the inner loop when we are creating the articles. this should increase speed quite a bit.

but, we can actually do one better. there is a bulk insert method using `insert_all` that lets us create a whole bunch of records at once without much overhead, and we can achieve that with a little bit of extra coding, now we have 3 loops, which isn't ideal, but that's what we get in this case.

```ruby
require 'csv'

def import_articles_with_tags(csv_file)
    tag_names_set = Set.new
    articles = []
    article_tags_insert = []

    CSV.foreach(csv_file, headers: true) do |row|
        tags = row['tags'].split(',').map(&:strip)
        tags.each{|tag_name| tag_names_set.add(tag_name)}
        articles << {title: row['title'], author: row['author']}

    end
    
    #fetch all tags or create them before creating articles.
    tags_hash = {}
    Tag.where(name: tag_names_set.to_a).each{ |tag| tags_hash[tag.name] = tag}
    tag_names_set.each do |tag_name|
        tags_hash[tag_name] = Tag.create!(name: tag_name) unless tags_hash[tag_name]
    end
    
    Article.insert_all(articles) # create all articles using bulk insert
    article_id_hash = {}
    Article.all.each{|art| article_id_hash[art.title] = art.id }# create a hash to get the article by title later


    CSV.foreach(csv_file, headers: true) do |row|
        tag_names = row['tags'].split(',').map(&:strip)
         tag_names.each do |tag_name|
            article_tags_insert << {article_id: article_id_hash[row['title']], tag_id: tags_hash[tag_name].id}
         end
    end
    ArticleTag.insert_all(article_tags_insert)
end
```

now we create a batch of articles and insert all of them at once using the `insert_all`, and after doing that we have to fetch the articles again to map their id’s and use them for inserting into the join table with the created tags. this is the most efficient way of processing that i know of with active record. of course you could use raw sql queries to speed it up even more, but at this point i would suggest if you are having issues with these methods, that maybe your csv file is too big and needs to be split before processing.

things can get even more complicated when you add validations, or if you need to handle more complex data transformations before creating your models, but i'd rather not get into that right now. it's almost friday, and i’m already thinking about that pizza i'm going to order after work, and frankly, a good pizza is much more complex to model than this example.

for further reading, i’d suggest checking out "patterns of enterprise application architecture" by martin fowler; it's a dense book, but has great insights into data mapping and relationships. also, looking at some papers about database indexing will help you understand better how databases store data, and how you can make these queries faster. if you want to go even further "database system concepts" by silberschatz is another great reference, though it goes way beyond what you would need here.

one last thing i want to point out, don't ever use `create!` without knowing what you're doing. it throws an exception if something goes wrong during creation, and while it's useful in development, you’ll probably want to handle errors more gracefully in production. so, use it wisely. i have crashed many server instances that way.
