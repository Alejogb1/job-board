---
title: "Why aren't friendly_id slugs working with Rails link_to helpers?"
date: "2024-12-23"
id: "why-arent-friendlyid-slugs-working-with-rails-linkto-helpers"
---

Alright,  I recall a particularly frustrating sprint a few years back where we ran headfirst into this exact issue—friendly_id slugs playing hide-and-seek with Rails' `link_to` helpers. We had meticulously set up friendly_id, generated slugs that looked perfectly valid in the database, and yet, the links were stubbornly defaulting to numeric ids. It was enough to make any developer question their sanity. So, what's going on under the hood, and how do we fix it?

The core of the problem stems from how Rails’ `link_to` helper determines the route and parameters for generating URLs. By default, it leverages Active Record’s `to_param` method. This method, without intervention, returns the primary key, usually the `id` of your model. Friendly_id, on the other hand, operates by creating a slug attribute (often `slug`) and associating it with a unique identifier for your records. The challenge arises because Rails' default routing logic isn't automatically aware of this 'slug' magic; it's still reaching for the `id`.

The most immediate solution, and the one most developers encounter first, is to override the `to_param` method within your model. This instructs Rails to use the slug instead of the id when generating URLs. However, there's a subtlety here that can trip you up if you're not careful. You don't want to just return the slug; you need to ensure it is compatible with how friendly_id expects to find the record. It's not enough to just return `self.slug`. You need to return something that `friendly_id` can use to identify the record, which in most cases is the slug itself.

Let’s look at a code example illustrating this. Suppose we have a model named `Article`. A basic implementation without overriding `to_param` would look like this:

```ruby
# app/models/article.rb
class Article < ApplicationRecord
  extend FriendlyId
  friendly_id :title, use: :slugged
end
```

Now, if you use `link_to` like so: `link_to 'View Article', @article`, Rails would generate a link using the `id` as a parameter, something like `/articles/123`. The crucial change is to override `to_param` like this:

```ruby
# app/models/article.rb
class Article < ApplicationRecord
  extend FriendlyId
  friendly_id :title, use: :slugged

  def to_param
    slug
  end
end
```

With this change, `link_to 'View Article', @article` now generates a URL like `/articles/my-article-title-slug`, which is exactly what we want. Friendly_id will then use the slug from the URL to find your `Article`. This is usually the first and most crucial step.

However, things can get a bit more complex in certain edge cases. Consider a situation where you have nested resources or want to include additional parameters in your routes. If your routes are not configured correctly or if you attempt to pass parameters that are not properly recognized by your application, the slugs might not work seamlessly with the generated links, even after overriding `to_param`. This is where route configurations in `config/routes.rb` become critical.

Let's say you have an `Article` belonging to a `Category`, and you need URLs like `/categories/category-slug/articles/article-slug`. The routes in `config/routes.rb` would look something like this:

```ruby
# config/routes.rb
Rails.application.routes.draw do
  resources :categories, param: :slug do
    resources :articles, param: :slug
  end
end
```

Notice the `param: :slug` within the `resources` block. This explicitly tells Rails to use the `slug` parameter in route generation for `Category` and `Article`. If we don’t include the `param: :slug`, the routes will expect an `id` and the generated url would be incorrect. It is imperative to set this for *all* resource routes in the application if you want to use slugs to identify records.

Here's an example where it matters:

```ruby
# app/controllers/articles_controller.rb
class ArticlesController < ApplicationController
  def show
    @category = Category.friendly.find(params[:category_slug])
    @article = @category.articles.friendly.find(params[:slug])
  end
end
```
If the routes are set correctly with `param: :slug`, the params will contain the slug value instead of an id, but the code in the controller will still correctly find the resources using the slug. Without the param option on routes, `params[:category_slug]` and `params[:slug]` would look for a category id and an article id respectively.
Finally, there are situations where you might need to implement a more nuanced `to_param` method. For example, when you have a localized application, you may wish to return the slug along with the locale in the url. In that case, your `to_param` might look more like this:

```ruby
# app/models/article.rb
class Article < ApplicationRecord
  extend FriendlyId
  friendly_id :title, use: :slugged

  def to_param
    "#{slug}-#{I18n.locale}"
  end

    def self.find_by_slug_and_locale(slug, locale)
      self.friendly.find("#{slug}-#{locale}")
    end
end
```
And here is how you might change your controller method to accommodate:
```ruby
# app/controllers/articles_controller.rb
class ArticlesController < ApplicationController
  def show
    @category = Category.friendly.find(params[:category_slug])
    slug, locale = params[:slug].split('-')
    I18n.locale = locale if locale
    @article = Article.find_by_slug_and_locale(slug, locale)
  end
end
```
This approach returns the slug with a locale component for a link such as `/articles/my-article-slug-en`, which is then parsed in the controller to load the appropriate record based on the slug and locale, which you might handle in a before action to set the appropriate locale for the session. This approach is a bit more complex, but illustrates that sometimes a more nuanced approach is required depending on the application’s requirements.

In summary, when you see friendly_id slugs failing to play nice with `link_to`, it almost always boils down to a few key areas: The overridden `to_param` method in your model, how you set the params in your routes using `param: :slug`, and how your application finds the records using the slug parameter in the controllers. Getting these three aspects working in harmony is paramount.

For further reading, I recommend "Agile Web Development with Rails 7" by Sam Ruby et al., which offers a comprehensive overview of routing and model interactions within the Rails ecosystem. Additionally, the official Rails Guides documentation is essential reading, particularly the sections on routing and Active Record. Finally, the source code for the friendly_id gem itself is extremely helpful; it's freely available on GitHub, and studying its implementation will enhance your understanding. Debugging this issue is often an exercise in carefully working backward from the view, through the routes, and into the model logic.
