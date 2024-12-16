---
title: "Why is Zeitwerk collapsing namespaces to the root level?"
date: "2024-12-16"
id: "why-is-zeitwerk-collapsing-namespaces-to-the-root-level"
---

Alright, let’s tackle this namespace collapse issue with Zeitwerk; it’s a puzzle I've certainly encountered in past projects, and it's not as straightforward as it initially seems. From my experience, the root cause isn't some inherent flaw in Zeitwerk itself but rather a particular interaction between how it infers namespaces and how your application's file structure is organized, particularly if you're not following its expected conventions strictly. The 'collapsing' effect where everything appears at the root level is actually a consequence of Zeitwerk not being able to identify the proper intended namespace from the directory structure and filename combination.

Zeitwerk operates on the principle of convention-over-configuration, which means it's designed to automatically determine the namespace based on file paths and class names. If the directory structure doesn't line up with the desired namespaces, or if there's a naming conflict, you’ll see that 'collapse'. For instance, it's very common for new Ruby on Rails developers or those migrating older projects to stumble upon this behavior. I recall one such situation a few years ago, working on a legacy project where models were initially nested deeper than the standard app/models structure. Zeitwerk, by default, will interpret files within, let's say, `app/models/legacy/users.rb` as belonging to the `Users` constant under the root namespace, unless explicitly told otherwise, instead of `Legacy::Users`. So instead of `Legacy::Users`, you might find a conflict with `::Users` if you've got a file directly in app/models like `app/models/users.rb`.

The key to understanding this is how Zeitwerk constructs its lookup map. It essentially maps directory segments to potential namespace fragments, and it expects a certain level of regularity in naming. When you have files outside its intended paths or with naming conventions that don't follow its expectations, it defaults to the root. This is not a bug but a failsafe, to prevent runtime errors arising from constant resolution issues. Zeitwerk favors safety over implicitly 'guessing' the namespace. I’ve seen codebases where developers attempted various "fixes" that only mask the underlying problem and introduce more technical debt in the long run. So avoiding such situations is paramount.

Let’s break this down further with some illustrative scenarios and code snippets.

**Example 1: Misaligned Directory Structure**

Imagine you have a file structure like this:

```
app/
  models/
    admin/
       users.rb
```

And your `users.rb` file contains:

```ruby
# app/models/admin/users.rb
class Users
  def self.list_all
     puts "Listing all admin users"
  end
end
```

In this scenario, by convention, Zeitwerk will load the class in `app/models/admin/users.rb` as `::Users` and not `Admin::Users`. If you intended to nest the class under the `Admin` module, you will not be able to access this using `Admin::Users.list_all` .

Here's a solution to correctly specify the namespace:

```ruby
# app/models/admin/users.rb
module Admin
  class Users
    def self.list_all
      puts "Listing all admin users"
    end
  end
end
```

Now Zeitwerk will load the class as `Admin::Users` because the `admin` directory has the associated module.

**Example 2: Naming Conflicts and Overlapping Paths**

Consider this file structure:

```
app/
  services/
      api/
         client.rb
  clients.rb
```

And contents of each file respectively:

```ruby
# app/services/api/client.rb
class Client
  def initialize(url)
    @url = url
  end

  def get
    puts "Fetching from api #{@url}"
  end
end

# app/clients.rb
class Client
  def initialize(name)
    @name = name
  end
  def greet
    puts "Hello #{@name} client"
  end
end
```

Here, you have a `Client` class in both `app/services/api` and directly under app as `app/clients.rb` . Zeitwerk will usually load the file by alphabetic order of the directory names in case there are clashes, leading potentially to unexpected behavior. You'll likely find that either `::Client` will be an instance of the `app/clients.rb` class or an error will be raised due to class name duplication.

The resolution involves ensuring namespacing by creating appropriate modules:

```ruby
# app/services/api/client.rb
module Api
  class Client
    def initialize(url)
      @url = url
    end

    def get
      puts "Fetching from api #{@url}"
    end
  end
end
```

And if `app/clients.rb` is intended to be root level we should rename it appropriately:
```ruby
# app/clients.rb
class AppClient
  def initialize(name)
    @name = name
  end
  def greet
    puts "Hello #{@name} client"
  end
end
```
Now you can correctly access your Api Client via `Api::Client` and your root level client via `AppClient`.

**Example 3: Explicit Namespace Declarations**

Sometimes, you might have a specific structure that doesn’t adhere to the typical Zeitwerk conventions. In such cases, you can explicitly instruct Zeitwerk how to manage namespaces, particularly through the use of a configuration. Although Zeitwerk's default behavior works well in many scenarios, it is useful to know how to alter its behavior for more complex scenarios.

Let's say you want `app/lib/workers/background_jobs.rb` to be in `Workers::BackgroundJobs` instead of just collapsing to `BackgroundJobs`:

```ruby
# app/lib/workers/background_jobs.rb
class BackgroundJobs
  def perform
    puts "Processing background job."
  end
end
```

And in your Zeitwerk configuration (for example, in `config/application.rb` or an initializer):
```ruby
# config/application.rb
module MyApplication
  class Application < Rails::Application
    config.autoload_paths << "#{root}/app/lib"

    config.zeitwerk.inflector = Zeitwerk::Inflector.new do |inflector|
      inflector.inflect(
        "workers" => "Workers",
       )
     end
  end
end
```
With this configuration, you are telling Zeitwerk to treat the folder `workers` as a module called `Workers` in the `app/lib` folder.

This example illustrates how to tweak Zeitwerk’s inflector to accommodate specific directory structures and override the default behavior where it would likely put `BackgroundJobs` into the root namespace.

**Recommended Resources**

For a deeper dive, I highly recommend consulting the following resources:

1. **The Zeitwerk Guide:** The official documentation on the Rails guides or the Zeitwerk repository on GitHub. It provides the most comprehensive details on how Zeitwerk works and how to configure it effectively.
2. **“Rails Autoloading with Zeitwerk” by Xavier Noria:** This presentation or article by one of the authors of Zeitwerk is invaluable for understanding the underlying mechanisms and rationale behind it.
3. **"Refactoring Ruby" by Martin Fowler and Kent Beck:** While not strictly Zeitwerk-specific, this book offers a deep understanding of object-oriented principles, which are foundational to understanding how namespaces operate in Ruby, particularly for designing more maintainable projects. The principles discussed assist when refactoring codebases to suit Zeitwerk.

In conclusion, the namespace collapsing in Zeitwerk isn’t an error but an indication that its convention-based namespace resolution isn't finding the structure it expects. By meticulously examining your file structure, understanding the conventions, and potentially employing explicit configuration when needed, you can successfully prevent these issues and leverage the efficient autoloading that Zeitwerk provides. It's a matter of understanding and applying its fundamental principles.
