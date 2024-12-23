---
title: "Why are Rails 7 `zichtwerk` constants not found?"
date: "2024-12-23"
id: "why-are-rails-7-zichtwerk-constants-not-found"
---

,  The issue of `zichtwerk` constants not being found in Rails 7, specifically, is something I encountered a couple of years back during a large-scale refactoring effort on a legacy Rails project. It's frustrating, to say the least, especially since the `zichtwerk` functionality is supposed to make constants readily available across the application. It points to a deeper understanding of how Rails autoloading, eager loading, and its interaction with constants—particularly within the zeitwerk system—works.

First, it's important to clarify that `zichtwerk` is the new, default code loading mechanism introduced in Rails 6 and heavily relied upon in Rails 7. Gone are the days of classic autoloading; `zichtwerk` is a more robust and deterministic solution, primarily operating under the principle of 'code first.' This means the constant's name directly corresponds to the file path in which the code defining it resides. If the file isn't where `zichtwerk` expects it, the constant simply won’t be found.

The most common culprit for constants not being found is a discrepancy in naming conventions and directory structure. `zichtwerk` expects files to follow a pattern: a `class` or `module` named `MyClass` is expected to reside in a file named `my_class.rb` under the correct namespace directory. For example, if you have a constant `MyApp::MyService` it is expected to be found in `app/services/my_app/my_service.rb`. Deviations from this, like incorrect casing or misplaced files, will almost always lead to a "NameError: uninitialized constant" error, which I've definitely spent far too much time debugging.

Another layer of complexity is added when we consider eager loading, which is crucial for applications in production. Eager loading works by pre-loading the application's constants during initialization, which prevents those annoying constant lookup errors during request execution. If a class or module wasn’t properly loaded during this phase, it won’t be available when referenced. This usually stems from situations where the code is in the project structure, but does not follow the naming and path conventions correctly, or is not under a directory that `zichtwerk` knows to look at.

Let’s illustrate this with some examples, drawing on experiences I had with that old Rails project of mine. We had a case where a constant `CustomLogger::Formatter` was not available, initially.

**Example 1: Incorrect File Naming**

Suppose we have the following project structure:

```
app/
  custom_logger/
    formatter.rb
```

And our `formatter.rb` file contains:

```ruby
# app/custom_logger/formatter.rb
module CustomLogger
  class Formatter
    def self.format(message)
      "[#{Time.now}] #{message}"
    end
  end
end
```

This setup *looks* correct but `zichtwerk` won’t find `CustomLogger::Formatter` because the file should be named `formatter.rb` *under* a folder called `custom_logger`, matching the module name. It is expecting a folder to mirror the module declaration of the class. To fix this, we would restructure the directories and have:

```
app/
  custom_logger/
    custom_logger/
        formatter.rb
```

The code will then correctly load with this setup.

**Example 2: Eager Loading Issues**

Now let’s say we have a model, `AuditLog`, within our `app/models` directory.  Let’s assume the file and directory structure are correct: `app/models/audit_log.rb`. The model’s code is:

```ruby
# app/models/audit_log.rb
class AuditLog < ApplicationRecord
  def self.log_action(action, user)
    puts "User #{user.name} performed action: #{action}"
  end
end
```

And we try to use this in our `app/controllers/users_controller.rb`.

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  def create
      @user = User.create(user_params)
      AuditLog.log_action("created user", @user) #This is an issue if not eager loaded
      redirect_to @user
  end
end
```

In development, everything *might* seem to work because of lazy loading, but in production environments with eager loading active, this would be an issue if `AuditLog` is not loaded when the server starts. The reason for this is because it's possible that `AuditLog` is not required by any other file at server start. To avoid this, we would need to ensure that the `AuditLog` class has been loaded during eager load phase. If the constant is loaded, everything will work, but if it is not loaded it will raise a `NameError` when `AuditLog` is referenced. In Rails 7, it’s likely that the model *would* be eagerly loaded under most situations, but if any custom configuration leads it to be excluded from eager loading, then it will fail in production. We can force eager loading of this by explicitly using it somewhere during the initial phase of our application.

**Example 3: Module Inclusion**

The issue also extends to included modules. If you have a module that uses constants and that module is not loaded, then those constants won't be available within the module. For example:

```ruby
# app/lib/my_module.rb
module MyModule
  def initialize(config)
     @config = config
     puts Config::API_KEY  # This will fail if config is not loaded first
  end
end

# config/initializers/configuration.rb
module Config
  API_KEY = 'abc123xyz'
end
```

And then we have a class that includes it:

```ruby
# app/services/my_service.rb
class MyService
  include MyModule
  def initialize
     super(options_from_environment)
  end
end

```

In this case, when `MyService` gets initialized, it will include `MyModule` and then within it it will attempt to access the `Config::API_KEY`, but this constant may not have loaded before `MyModule` does, and therefore raise an error. The fix here would be to ensure the constants are loaded early on or that a suitable way of getting the data exists, perhaps through a class variable, or environment variable.

Debugging these issues often involves a careful review of file paths, module/class names, and an understanding of where your constants are being utilized. I’ve found that a combination of the rails console (`rails console`), and careful use of debugging (`byebug` or `pry`) at various stages of startup is essential. Setting breakpoints right before the constant is accessed can often give insight into whether the constant was ever loaded or not. Also, pay close attention to `config/application.rb`, where you configure things such as `eager_load_paths` since these are crucial for specifying which directories are checked by the framework for code during boot.

For a deep understanding of this, I’d recommend reading the official Rails guides, specifically the sections on autoloading, eager loading, and `zichtwerk`. There’s a good chapter on `zichtwerk` in *Crafting Rails Applications* by José Valim and a related discussion in the *Rails 7 Upgrading Guide*. It’s also beneficial to examine the source code of the `activesupport` gem, where the core `zichtwerk` logic resides. Spending time getting comfortable with the framework's internal workings can significantly reduce these sorts of problems in the long run.

In summary, `zichtwerk`'s strict naming conventions are a double-edged sword. They make the code loading process highly predictable and deterministic, but any deviations can lead to unexpected errors. By understanding the underlying mechanisms and paying attention to directory structures and naming, you can avoid the frustration of constants not being found, and I assure you, it’s an area that’s well worth investing some time in. The more familiar you are with `zichtwerk`, the faster you can resolve these issues.
