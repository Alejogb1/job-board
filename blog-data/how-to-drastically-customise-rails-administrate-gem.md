---
title: "How to drastically customise Rails Administrate gem?"
date: "2024-12-14"
id: "how-to-drastically-customise-rails-administrate-gem"
---

alright, so you're looking to really bend administrate to your will, huh? i get it. it's a great gem, gets you up and running with a decent admin interface super quickly, but sometimes that 'decent' just doesn't cut it. i’ve been down this road myself, multiple times. let me share some of the pain, and hopefully some solutions, that i've learned the hard way.

my first real encounter with administrate's limitations was back when i was working on a system for managing research papers, i wanted to display the pdfs in a fancy viewer inline, rather than just linking to them. administrate out of the box gives you a simple link, which felt so outdated, i mean, come on. initially, i tried some crazy javascript hacks, patching things in after the page loaded, a nightmare. the maintenance was unbearable, each update of the gem broke half of my modifications. lesson learned, don't be that guy. so, yeah, i feel your pain. lets tackle this properly.

administrate uses a pretty straightforward structure, controllers, views, and forms, so the key to customization is understanding how these pieces fit together.

first thing, forget about messing with the gem's files directly. don't even think about it. it’s a recipe for disaster down the line with updates. the official recommendation is to use decorators, and they are your best friend here, they let you tweak how attributes are displayed and how forms behave.

let's start with the easy stuff, changing how attributes appear. imagine your `article` model has a `body` field which stores markdown, you want this to be rendered as html in the admin panel instead of just plaintext. here's a simple decorator example, the magic lies in defining `attribute` methods, so it knows how to handle it:

```ruby
class ArticleDashboard < Administrate::BaseDashboard
  ATTRIBUTE_TYPES = {
    id: Field::Number,
    title: Field::String,
    body: Field::Text,
    created_at: Field::DateTime,
    updated_at: Field::DateTime
  }.freeze

  def display_resource(article)
    article.title
  end
end

class ArticleDecorator < Administrate::Decorator
  def body
    helpers.markdown(super) # 'super' gets the original value, then pipe it through the markdown helper
  end
end
```

note that you need to create a helper method called `markdown` in your `application_helper.rb`, and include this line in the decorator: `include ActionView::Helpers::TextHelper`, to make it work. you will also need to use a library like redcarpet to handle markdown parsing in the helper.

this gives you the power to reformat any attribute and you can add your own logic before rendering it. if you want to modify the form fields, you can do that as well, but it requires a little more effort. for this, you are going to create a custom field. let's say you are tired of the standard text area for the body, and want a nice wysiwyg editor instead.

here’s an example how you would go about that, you can create a custom field component in the `app/fields/` directory:

```ruby
# app/fields/wysiwyg_field.rb
require "administrate/field/base"

class Field::Wysiwyg < Administrate::Field::Base
  def to_s
    data # it stores data in the data variable of the field
  end

  def render_form
    form_builder.text_area(attribute, { class: "wysiwyg" })
  end
end
```

now, include this field in your dashboard, like so:

```ruby
class ArticleDashboard < Administrate::BaseDashboard
  ATTRIBUTE_TYPES = {
    id: Field::Number,
    title: Field::String,
    body: Field::Wysiwyg,  # using the new custom field here
    created_at: Field::DateTime,
    updated_at: Field::DateTime
  }.freeze

  def display_resource(article)
    article.title
  end
end

class ArticleDecorator < Administrate::Decorator
  def body
    helpers.markdown(super)
  end
end
```

note that you will need to add a javascript library to enable your wysiwyg editor. i would recommend using a solid library like 'tinymce' because it's lightweight and has tons of useful features. you can also include a stylesheet to make it look pretty. this kind of customization gives you granular control over the form and display. remember to add `javascript_pack_tag` and `stylesheet_pack_tag` in your layout to load your assets.

if you wanna really go off the deep end, you'll need to start customizing the actual controllers. the admin controller that you modify, inherits from `Administrate::ApplicationController` and you can override the methods to tweak behavior. i recommend keeping your changes small if you do this. for example, if you need to add a parameter before creating a new record you can do this:

```ruby
# app/controllers/admin/articles_controller.rb
class Admin::ArticlesController < Administrate::ApplicationController
  def create
    params[:article][:author_id] = current_user.id
    super
  end
end
```

note here that you must create a folder in `app/controllers/admin/` with a controller named exactly like the one you are trying to override, and you should make it inherit from the administrate controller. this gives you control over things like strong parameters, authorization, and how records are handled.

i also noticed there is no easy way to use a custom query in the dashboard, for example, if you want to show only active articles in the index page. there is no configuration for that. and you have to customize the index action in the controller. this is how you do it:

```ruby
# app/controllers/admin/articles_controller.rb
class Admin::ArticlesController < Administrate::ApplicationController
  def index
    search_term = params[:search].to_s.strip
    resources = Article.where(active: true).where("title like ?", "%#{search_term}%").page(params[:page]).per(records_per_page)
    page = Administrate::Page::Collection.new(dashboard, order: order, resources: resources, search_term: search_term)
    render :index, locals: { page: page, search_term: search_term, resources: resources, show_search_bar: show_search_bar? }
  end
end
```
note that `records_per_page` is a constant that can be defined inside of your custom controller.

now, about resources, i highly recommend checking out "patterns of enterprise application architecture" by martin fowler, it’s a must-read for understanding how to structure large applications. and if you are not familiar with ruby, "programming ruby" by dave thomas, andy hunt, and chad fowler can help you with more fundamental ruby concepts. it explains the basics of ruby metaprogramming. administrate makes extensive use of this.

if your problems are mainly styling you might consider changing the stylesheet. and i recommend reading the official rails guides carefully and understanding the asset pipeline, because you will probably need to tweak some things to add your own stylesheets and javascript to customize the administration panel.

so, there you have it. a slightly opinionated but hopefully helpful overview of how to customize administrate. remember, the key is to use the extension points the gem provides. avoid monkey patching whenever possible, always try to find an appropriate way to extend the base functionality with decorators or custom fields. keep it simple, keep it maintainable. and remember, if you think your code is too complex, it probably is.
