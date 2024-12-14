---
title: "When I run a migration, why is config.assets not working in rails 7?"
date: "2024-12-14"
id: "when-i-run-a-migration-why-is-configassets-not-working-in-rails-7"
---

alright, so you're hitting a classic rails 7 assets pipeline gotcha, i've definitely been there, felt that pain. it's not that the config.assets isn't *working* per se, it’s more about how rails 7 handles asset compilation and how it interacts with your `config/initializers/assets.rb` file. let's get into it.

the shift in rails 7 is significant in that it leans very heavily into the esbuild/importmaps/tailwindcss model, especially for new apps. the old sprockets-based asset pipeline which is where `config.assets` was heavily utilized is still there, but it's definitely not the primary focus anymore. previously, sprockets would compile all your assets (css, js, images, etc.) into a single directory (`public/assets`) and rails could then serve them from there. your `config.assets` directives were crucial for that.

now, when you create a new rails 7 app or migrate to it, you may find that sprockets isn't even enabled by default. you could be using the jsbundling-rails and cssbundling-rails gems. if you are, that means you are now dealing with a setup where you need to understand how those gems influence your asset workflows. if you are using the gem `sass-rails` is still relevant and that can be confusing.

the problem you are seeing often comes down to a few common scenarios. lets go through a couple of possible cases:

**scenario 1: sprockets isn't enabled or is bypassed for specific assets**

if you aren't using sprockets actively or if you have a specific asset that is included in esbuild or importmaps. you might have something like this in `config/initializers/assets.rb`:

```ruby
# config/initializers/assets.rb
Rails.application.config.assets.precompile += %w( admin.css )
Rails.application.config.assets.paths << Rails.root.join('app', 'assets', 'fonts')
```

this might look correct but if you have a file located under `app/assets/stylesheets/admin.css` rails will not pick it up with the normal command `rails assets:precompile` as expected. the sprockets pipeline may not be responsible for compiling `admin.css` if the file is being processed through the javascript/css bundling gems. if you try to precompile it, it will not appear on `public/assets` as expected. also, if you had fonts inside `app/assets/fonts` they also may not be picked up by this configuration, and that can be frustrating if you did not expect this behaviour.

to solve this, and keep using your `admin.css` file you must add a line to your `application.css` file like this:

```css
/* app/assets/stylesheets/application.css */
*= require admin
```

and now it will be properly compiled via sprockets if it is configured as your css bundling mechanism.

**scenario 2: confusion between manifest.js and asset.rb**

another common source of confusion is the relationship between `config/initializers/assets.rb` and your `app/assets/config/manifest.js`. if you are using importmaps or a javascript bundler, rails will use `manifest.js` to define which assets to load. this can get tricky if you are also trying to use `config.assets` in the old way. the `manifest.js` takes precedence for javascript and css when using the bundlers, and can lead to you adding assets to `assets.rb` that will never be compiled with your application. it is an important distinction to understand the difference between the two configuration files to avoid this problem. if you add a file in `assets.rb` that has the same name as a file described in `manifest.js` you are creating a conflict.

example of `manifest.js` file:

```js
//= link_tree ../images
//= link_directory ../stylesheets .css
//= link_tree ../../javascript .js
//= link_tree ../../../vendor/javascript .js
```

**scenario 3: development vs. production issues**

also, it's worth mentioning that asset behaviour can differ between development and production environments. in development, rails typically serves assets on demand, while in production, precompilation is expected. this can cause issues when you test your application in development, things seem correct, and then it breaks in production when the assets are missing or loaded in the wrong way. you might see your changes not being picked up. this has happen to me in the past more than i want to.

this is a typical situation that happens when you have different configurations for assets in development and production.

here is a very simple case on how to have a specific asset configured using the rails way, you would configure this inside the file `config/environments/production.rb`

```ruby
# config/environments/production.rb
Rails.application.configure do
    config.assets.precompile += %w(print.css)
end
```

and you would have a separate file to be used on development inside `config/environments/development.rb`:

```ruby
# config/environments/development.rb
Rails.application.configure do
    config.assets.precompile += %w(debug.css)
end
```

to make your application work, in development and production, you will have to create the files under `app/assets/stylesheets` directory.

```
app/assets/stylesheets/print.css
app/assets/stylesheets/debug.css
```

**some practical advice and how i fixed it in the past**

so, what can you do to fix this? the first thing is to figure out what is your asset management strategy is. is it `sprockets`, or `jsbundling-rails`/`cssbundling-rails`? if you are using jsbundling/cssbundling, then you must update `manifest.js` and not `assets.rb` to add or remove new assets. this is the most common mistake when migrating to rails 7 in my experience. i made that mistake and it took me a couple of hours to understand what i was doing wrong.

if you decide to keep using sprockets as you asset management strategy for rails 7 then you need to ensure that sprockets is enabled. you can enable it by adding the following line to your `Gemfile`:

```ruby
# Gemfile
gem 'sprockets-rails'
```

after adding the sprockets gem to your `Gemfile`, you should also enable it in your `config/application.rb` file:

```ruby
# config/application.rb
module YourAppName
  class Application < Rails::Application
    config.load_defaults 7.0

    config.assets.enabled = true
  end
end
```

you should also add `require "sprockets/railtie"` to your `config/application.rb` like this:

```ruby
require_relative "boot"
require "rails/all"
require "sprockets/railtie"
```

after this, your `assets.rb` file and the commands `rails assets:precompile` or `rails assets:clobber` should behave as you would expect. if you are using a more modern javascript/css bundling approach, you would be better off by removing the `config.assets` configuration from the file `assets.rb` and using your `manifest.js` for the configuration.

and here is the joke: why did the rails developer break up with the sprockets? because they had too many dependency issues. (i know, not my best work)

**resources that helped me when i started with rails 7**

as for resources, i would highly recommend going through the official rails guides (they've improved a lot), specifically the sections on the asset pipeline and javascript/css bundling. for a deeper understanding on sprockets, you should take a look into the sprockets documentation. a book about advanced rails programming like "agile web development with rails 7" or "rails 7 in action" can be useful as well. they usually go deeper into the details of assets.

finally, the rails source code itself is invaluable. you can navigate the code base to see exactly what is going on under the hood, and this has help me a lot. just look at the rails github repository for the asset pipeline implementation.

i really hope this helps you avoid similar headaches that i had in the past. assets in rails can be tricky, especially when migrating between versions. if you have any more questions, please let me know.
