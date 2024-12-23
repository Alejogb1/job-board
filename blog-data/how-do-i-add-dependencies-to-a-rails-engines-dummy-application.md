---
title: "How do I add dependencies to a Rails engine's dummy application?"
date: "2024-12-23"
id: "how-do-i-add-dependencies-to-a-rails-engines-dummy-application"
---

,  Been around the block a few times with Rails engines and their often-peculiar needs, particularly when it comes to dependencies in that dummy application. It’s a common stumbling block, and getting it smooth often requires understanding the quirks of Rails' architecture, rather than just blindly following tutorials. I've seen, and personally debugged, more than my fair share of dependency-related headaches in this context, so let’s get into the specifics.

The core issue here stems from the fact that a Rails engine is, fundamentally, a mini-Rails application embedded within a larger host application. The `test/dummy` application is essentially an even *smaller* Rails application used exclusively for testing and development of your engine. It doesn't inherit dependencies automatically; they need to be declared explicitly. This ensures the engine's test suite is isolated and doesn’t inadvertently pull in dependencies from the host that it isn't designed to handle.

Think about it like this: your engine provides a specific set of functionalities. The `test/dummy` app acts like a sandbox environment to test those specific features in isolation, without the noise of the host application. That means we need to declare all of the gem and asset dependencies that the engine depends upon in order for the dummy application to successfully run tests and perform development activities.

The typical gotcha is assuming the dummy app will automatically pick up the dependencies declared in the engine's main `Gemfile`. That’s a no-go. Instead, the dummy application, residing within the `/test/dummy` folder, has its *own* `Gemfile`, and potentially, its own dependency management needs for assets, javascript files etc. This is where we need to focus our attention.

Here's the breakdown of how to properly manage this, illustrated with practical code examples:

**1. The Gemfile in `test/dummy`:**

First, and perhaps most obvious, is the `Gemfile`. Your primary engine's `Gemfile` declares the gems required for the core engine itself. We need to repeat, or add to, those same dependencies into the dummy application's `Gemfile`.

Here's how it looks:

```ruby
# test/dummy/Gemfile

source 'https://rubygems.org'

gem 'rails', '~> 7.0' # Or whatever version your host app uses
gem 'sqlite3'
gem 'pg' if ENV['DATABASE_URL'] # Example conditional dependency

gem 'rspec-rails' # Example engine test dependency

# Dependencies required by the engine itself
gem 'some_gem_from_engine'
gem 'another_gem_needed'
```

This `Gemfile` should mirror the dependencies declared in the engine's root `Gemfile`, specifically those the dummy application will need for the test suite to function. Note, it's not a cut-and-paste job; you might add additional testing dependencies that are *not* required by the parent engine. Always err on the side of explicitness.

After modifying this `Gemfile`, run the following command from inside the `test/dummy` directory to ensure all of those dependencies are correctly installed:

```bash
bundle install
```

**2. Adding JavaScript/Asset Dependencies:**

Sometimes your engine or the dummy app also requires JavaScript libraries or other assets. This is often managed through the asset pipeline or, more recently, tools like webpack or importmaps.

Here’s how to incorporate those dependencies within your dummy application:

**a) Asset Pipeline (`app/assets/javascripts/application.js` and related folders):**

 If you still use sprockets, in your `test/dummy/app/assets/javascripts/application.js` file, ensure you are requiring the needed files:

```javascript
//= require rails-ujs
//= require activestorage
//= require jquery # Example dependency
//= require_tree .
```

This file is directly responsible for compiling all of your application specific JavaScript assets.

**b) Importmaps or Webpacker (or other bundling solutions):**

If you’re using modern approaches like importmaps or webpack, your setup will be slightly different. For importmaps, you’d likely have a dedicated `config/importmap.rb` file in `test/dummy`. For webpacker, a `config/webpack/webpack.config.js` file. You need to ensure the packages required by your engine are installed in the dummy application's node_modules and that they're properly configured within your webpack or importmap setup.

For example with importmaps, you might have:

```ruby
# test/dummy/config/importmap.rb

pin "application", preload: true
pin "@hotwired/turbo", to: "turbo.min.js", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true

pin "jquery", to: "https://ga.jspm.io/npm:jquery@3.7.0/dist/jquery.js"  # Example
pin "your_engine_js", to: "your_engine_js.js"  # Add a relative path, if necessary.
```

Ensure that these pinned imports accurately reflect your engine's Javascript requirements and use the correct pathing. I've spent many hours tracking down incorrect paths, so it’s a good practice to double-check.

After updating your importmaps, ensure you run:

```bash
bin/rails importmap:install
```
or the appropriate webpack command (e.g. `yarn install` or `npm install`) to sync the node dependencies.

**3. Database Migrations:**

A less obvious point, but just as critical. The dummy application might need specific database migrations, especially if your engine defines its own models or uses a database.

Here's the standard approach:

```ruby
# test/dummy/db/migrate/20231027120000_create_some_tables.rb

class CreateSomeTables < ActiveRecord::Migration[7.0]
  def change
    create_table :some_tables do |t|
      t.string :name
      t.timestamps
    end
  end
end
```

You will need to run those migrations in your dummy database once they've been written:

```bash
bin/rails db:migrate
```

This ensures that the tables required by your engine exist in the dummy application's testing environment.

**Key Takeaways & Recommended Reading:**

*   **Specificity is key:** The dummy app's `Gemfile`, asset configuration, and migrations are *separate* from the engine's and the host application. Treat it as its own little application when it comes to dependency management.

*   **Debugging:** If you encounter weird errors (like "uninitialized constant" or javascript that isn't properly compiling), check the dummy app's `Gemfile` and asset pipeline/bundling configurations first. Often it's a missing dependency here.

*   **Version Consistency:** Make sure your gem versions match between the host app and dummy app where appropriate. Mismatched versions can cause strange, hard-to-debug behavior.

For further in-depth knowledge, I recommend reading these resources:

*   **"Crafting Rails Applications" by José Valim:** This book offers a detailed explanation of Rails architecture and its nuances, including the engine system.
*   **The official Rails documentation:** The section on engines provides the official guide to setting up engines and their dependencies. It's always worthwhile to refer to the official documentation.
*   **"Rails 7 API Documentation" by The Rails Core Team:** The official API docs are detailed and will offer helpful insight on all the specific classes and modules that you might be having issues with.

In summary, adding dependencies to a Rails engine's dummy application is a manual process that requires careful attention to detail. It involves mirroring or adding necessary dependencies in the dummy application’s own `Gemfile`, configuring asset pipelines or bundlers correctly, and applying relevant database migrations. By explicitly managing these dependencies, you ensure that your engine's tests are isolated and reliable, leading to a more robust and maintainable code base. It’s one of those things that seems straightforward, but often the devil is in the details of a few misconfigured files. Trust me; I’ve been there.
