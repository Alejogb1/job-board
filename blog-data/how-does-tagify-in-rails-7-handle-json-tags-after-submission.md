---
title: "How does Tagify in Rails 7 handle JSON tags after submission?"
date: "2024-12-16"
id: "how-does-tagify-in-rails-7-handle-json-tags-after-submission"
---

Alright, let's talk about how Tagify plays with JSON tags in a Rails 7 context, a topic I've certainly spent my fair share of time navigating. I remember back in my days working on that community forum project, we adopted Tagify for a richer tagging experience, and subsequently had to deep dive into how Rails handled the submitted data, particularly the JSON aspect. Let me lay out the mechanics from my perspective, drawing from that experience, and provide some tangible code samples to solidify the concepts.

First, let's set the stage. Tagify, on the frontend, usually generates a structured array of tags, often represented as JSON objects, like this: `[{value: "technology"}, {value: "programming"}, {value: "rails"}]`. When this data is submitted within a form, typically via a POST request, it's important to understand the various steps Rails takes to process it. Now, the core thing here is that Rails, by default, doesn't automatically convert a complex JSON structure nested within form data into something that's directly usable by your model. It treats that JSON string as just that – a string.

This is where you, the developer, need to intervene to properly parse and persist this data. We're not dealing with a simple key-value pair; it’s a structured array that needs to be processed. My team, at the time, quickly realized this wasn't a magical process that Rails handled implicitly.

Now, let's walk through what generally happens. When you submit a form with Tagify data, the data comes to your Rails controller as part of the `params` hash. Assuming your Tagify field is named, for instance, `tag_names`, your controller `params` will contain a key called `tag_names` and its corresponding value which would be a JSON string.

Here’s where the first piece of code comes in. We'll parse the JSON string into a Ruby array of hashes, before the model layer. Within the controller, you will normally have a method called something along the lines of `create` or `update`, so let’s look at an example:

```ruby
class ArticlesController < ApplicationController
  def create
    @article = Article.new(article_params)

    if @article.save
      redirect_to @article, notice: 'Article was successfully created.'
    else
      render :new, status: :unprocessable_entity
    end
  end

  private

  def article_params
    tags_param = params[:article][:tag_names]
    parsed_tags = tags_param.present? ? JSON.parse(tags_param) : []

    params.require(:article).permit(:title, :content).merge(
      tag_names: parsed_tags.map { |tag| tag['value'] }
    )
  end
end
```

In this first example, we use `JSON.parse` to convert the string into a Ruby array of hashes, each representing a tag. Subsequently, we map this array to extract the `value` of each tag, effectively preparing a simple array of strings that our `Article` model can handle. This is a crucial first step. I’ve seen scenarios where developers forget this step and end up with a string of json in their database – definitely something to avoid!

Now, how do you handle this parsed array in your model? There are many approaches. The most typical would be to create a relationship with another `Tag` model. Let's move onto the model layer, where this array of tag names can be used to manage database records. This involves either creating new tags or using existing ones.

Here's the second code snippet, demonstrating this functionality within our `Article` model:

```ruby
class Article < ApplicationRecord
  has_many :article_tags
  has_many :tags, through: :article_tags

  def tag_names=(names)
    self.tags = names.map { |name| Tag.find_or_create_by(name: name) }
  end
end
```

Here, we define a setter method `tag_names=` which takes the array of tag names, and utilizes `find_or_create_by` to find existing tags, or if they don’t already exist, to create new ones, associating them with the article. This ensures you don’t end up with duplicate tags in the database. I recall in my past experience seeing multiple variations of this implementation but the core premise remained consistent. The goal was always to convert those Tagify json-strings into associated database models.

In case you're dealing with a more complex data structure from Tagify, perhaps including metadata beyond just `value`, you'd need to adjust your parsing and model logic accordingly. For instance, Tagify allows adding properties like `locked`, `readonly` and so on. The `JSON.parse` step would need to handle those as well. Let’s take a slightly expanded example where tags also have a `locked` property:

```ruby
class ArticlesController < ApplicationController
  def update
    @article = Article.find(params[:id])

    if @article.update(article_params)
      redirect_to @article, notice: 'Article was successfully updated.'
    else
      render :edit, status: :unprocessable_entity
    end
  end

  private

  def article_params
    tags_param = params[:article][:tag_names]
    parsed_tags = tags_param.present? ? JSON.parse(tags_param) : []

     params.require(:article).permit(:title, :content).merge(
      tags_attributes: parsed_tags.map.with_index { |tag, index| {
        id: (Tag.find_by(name: tag['value'])&.id),
        name: tag['value'],
        locked: tag['locked'],
        _destroy: (tag['locked'] == true) ? '0' : '1'
      }.compact }
    )
  end
end
```

```ruby
class Article < ApplicationRecord
  has_many :article_tags
  has_many :tags, through: :article_tags
  accepts_nested_attributes_for :tags, allow_destroy: true

    def tag_names=(names)
        # no more implemented logic for direct association
        # nested_attributes takes care of associations
    end

end

```

In this case we're accepting nested attributes, and are able to pass parameters directly to our tags, additionally we've included a `_destroy` attribute which will remove an item if `locked` is true. You’d similarly adjust the `tag_names=` method in your model to reflect this new structure.

Finally, after processing all the tags, be it via simple name associations or via more complex nested attributes, you’ll need to decide how to use these processed tags, if you want to display them as well, for example in the `show.html.erb` template. That's usually straightforward: You’d loop through `@article.tags` and render the tag information accordingly.

For anyone wanting to deep dive into topics covered, I would highly recommend checking out "Agile Web Development with Rails 7" by Sam Ruby et al. It provides a comprehensive understanding of how rails works and its patterns for interacting with data. Additionally, for a solid understanding of form handling in rails, you should consult the official Rails guides. They offer a deep dive into parameters, model associations, and nested attributes. Further, for more specific information on JSON handling, the standard library documentation for Ruby’s JSON module provides technical details you might find relevant.

In summary, while Tagify provides a powerful frontend experience for managing tags, it's crucial to understand how this data is handled by Rails on the backend. Remember that JSON string needs to be parsed, and data needs to be organized before being consumed by your Rails models. The provided examples should offer a firm base for tackling such scenarios and hopefully help future developers as they journey to more advanced implementations.
