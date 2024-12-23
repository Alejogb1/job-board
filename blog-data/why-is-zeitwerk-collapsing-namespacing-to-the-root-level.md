---
title: "Why is Zeitwerk collapsing namespacing to the root level?"
date: "2024-12-23"
id: "why-is-zeitwerk-collapsing-namespacing-to-the-root-level"
---

Alright, let's delve into this Zeitwerk namespace collapsing issue. It’s something I’ve encountered myself, not just in toy projects, but in fairly substantial Rails applications where proper autoloading is paramount. The headache of namespace collisions appearing at the root level due to unexpected Zeitwerk behavior can be quite frustrating if you're not familiar with the nuances. I recall one particular incident where we were migrating an older application to a modern rails setup, and the namespace collapsing initially brought development to a standstill. We spent a good day or so ironing it out, and I've since spent enough time working with Zeitwerk to understand the core reasons behind this behavior, and how to effectively manage it.

The central issue arises from how Zeitwerk interprets file system paths and their associated namespaces. Essentially, it relies on conventions—specifically the structure of your application’s directories and files—to determine the corresponding Ruby namespaces. If these conventions aren't adhered to, Zeitwerk can get confused, leading to namespaces being incorrectly loaded, often collapsing them to the root level.

Here's a breakdown of the primary reasons I've observed why this happens:

1.  **Misplaced Files:** This is probably the most common culprit. Zeitwerk expects files within a directory to represent a namespace with the directory’s name, but only if the directory itself is within one of the configured autoload paths (those in `config.autoload_paths` in your `application.rb` or a similar initializer). If, for instance, you have a file `app/services/legacy_code/my_processor.rb` and your autoload paths *only* include `app/models`, then Zeitwerk *won't* load it under the `LegacyCode` namespace. Instead, since Zeitwerk cannot see a root-level `services/legacy_code` directory, it effectively treats this `my_processor.rb` as directly under your application's root module, leading to a conflict and namespace collapse. It is effectively seen as `::MyProcessor`.

2.  **Non-Conventional Directory Structures:** Zeitwerk heavily leans on the assumption that folder names directly translate to module names. Deviations from this practice cause problems. If you have a directory named `legacy-code` instead of `legacy_code`, Zeitwerk will not automatically understand that it should correspond to `LegacyCode`, and you might experience namespace collisions, or the files might simply not be autoloaded correctly. It is *vital* to maintain the consistent naming scheme required by zeitwerk, otherwise, things will break badly, quickly.

3.  **Ambiguous File Names:** Similar to directories, filenames are significant. Suppose you have a file named `my_model.rb` within `app/models`. Zeitwerk would typically assume it to define a `MyModel` class. But, if you have another file named `my_model.rb` within, say, a directory that isn’t a recognized namespace (like `app/not_namespace`), zeitwerk might get confused or, worse case, it might load both at the root level, overwriting each other or producing unexpected results. It’s essential to ensure that all file names within your code base are unique and within their correct namespaces as this is also what Zeitwerk uses to construct the appropriate constant names.

Now, let's see how these concepts manifest in actual code with examples and how you can avoid them:

**Example 1: The Misplaced File Issue**

Assume you have an app where you create a file, and we will demonstrate how this can lead to namespace issues, this is not an unusual problem with legacy code when not using the conventional location structure.

```ruby
# In config/application.rb (or similar initializer)
config.autoload_paths << "#{Rails.root}/app/models"
# But NOT including config.autoload_paths << "#{Rails.root}/app/services"

# Then you have
# app/services/legacy_code/my_processor.rb

module LegacyCode
  class MyProcessor
      def process(data)
         puts "Processed #{data}"
      end
  end
end

# And in your code you have
 LegacyCode::MyProcessor.new.process("some data")
```
Here, Zeitwerk is configured to autoload only from `app/models`. It's *not* looking at `app/services`. Consequently, when you reference `LegacyCode::MyProcessor`, Zeitwerk fails to find a `LegacyCode` module via the autoload paths and it loads `my_processor.rb` into the root namespace (in this instance, `::LegacyCode` as opposed to what we expect within our application's module). That is, because it sees a constant named `LegacyCode` inside the file, it will load that into the global scope, potentially conflicting with any pre-existing constants or even other namespaces that you have under the root scope. This is, most definitely, a problematic behavior.

**Solution:** You need to include the correct folder in your autoload path as follows:
```ruby
# In config/application.rb (or similar initializer)
config.autoload_paths << "#{Rails.root}/app/models"
config.autoload_paths << "#{Rails.root}/app/services"
```

This ensures Zeitwerk knows where to look and correctly associates `app/services/legacy_code/my_processor.rb` with `LegacyCode::MyProcessor`. This resolves the issue of loading the module at the root level.

**Example 2: Non-Conventional Directory Names**

Suppose you have a directory named using hyphens, which is not the standard folder naming convention.

```ruby
# In config/application.rb
config.autoload_paths << "#{Rails.root}/app/legacy-tools"

# Then you have the following, note the use of - in place of _, which is the convention
# app/legacy-tools/data_handler.rb

module LegacyTools
  class DataHandler
      def handle(data)
          puts "Handling #{data}"
      end
  end
end

# Then you have the following in your other code
LegacyTools::DataHandler.new.handle("some other data")
```

In this example, since Zeitwerk sees `app/legacy-tools`, it will attempt to map that to a module `LegacyTools`, which won't work, and therefore might result in the class `DataHandler` either not loading correctly or being loaded at the root level. Zeitwerk’s reliance on naming conventions is key here.

**Solution:** Change your directory naming convention to underscores:
```ruby
# In config/application.rb
config.autoload_paths << "#{Rails.root}/app/legacy_tools"

# rename app/legacy-tools to app/legacy_tools

# Now the file app/legacy_tools/data_handler.rb will be autoloaded as expected under LegacyTools
```

Adhering to the convention of using underscores rather than hyphens is absolutely key.

**Example 3: Ambiguous Filenames**

Imagine the issue caused by duplicate filenames, this is something that I saw cause some significant problems in a production application when moving from an older to a newer rails setup.
```ruby
# config/application.rb
config.autoload_paths << "#{Rails.root}/app/models"
config.autoload_paths << "#{Rails.root}/app/helpers"
# app/models/my_module.rb
class MyModule
    def module_operation
        puts "module operation 1"
    end
end
# app/helpers/my_module.rb
class MyModule
   def module_operation
    puts "module operation 2"
   end
end
```
In this, we have two files with the same filename, `my_module.rb`, in two different directories that *are* autoloaded. When this occurs, you will have race conditions, and overwriting, leading to instability in your code.

**Solution:** Be explicit in your file names, so the name is descriptive and unique:
```ruby
# config/application.rb
config.autoload_paths << "#{Rails.root}/app/models"
config.autoload_paths << "#{Rails.root}/app/helpers"

# app/models/my_data_model.rb
class MyDataModel
    def module_operation
        puts "module operation 1"
    end
end
# app/helpers/my_module_helper.rb
class MyModuleHelper
   def module_operation
    puts "module operation 2"
   end
end
```

By ensuring the names are more specific, we can prevent this namespace collision and overwrite.

**Recommendations for Further Study**

For a more in-depth understanding, I highly recommend these resources:

1.  **Rails Guides on Autoloading:** Specifically, the guides relating to Zeitwerk are a must. It’s important to have a solid grasp of the official documentation. This explains the mechanisms behind Zeitwerk’s behavior.

2.  **"Metaprogramming Ruby" by Paolo Perrotta:** This book, although older, provides a fantastic explanation of how Ruby namespaces work under the hood. It helps understanding the basics that zeitwerk relies on.

3.  **The Zeitwerk gem documentation on GitHub:** This repository includes detailed documentation and examples, that is usually kept fairly up to date and helps to understand the latest changes and specific edge cases.

In summary, namespace collapsing to the root level is usually the consequence of violating Zeitwerk’s expectations, especially regarding file placement, directory structure, and file naming. Always ensuring your paths are correctly specified, adhering to conventions for directory naming, and maintaining uniqueness in your filenames usually avoids these types of issues. It's a common pain point when working with rails, but these simple rules will make your life a lot easier.
