---
title: "How to resolve a ModuleNotFoundError when deploying a Rails app to Heroku?"
date: "2024-12-23"
id: "how-to-resolve-a-modulenotfounderror-when-deploying-a-rails-app-to-heroku"
---

Okay, let’s tackle this. I’ve seen this `ModuleNotFoundError` rear its head countless times over the years, particularly with Rails deployments to Heroku. It’s frustrating, I get it, especially when everything seems to work fine locally. The core issue, more often than not, revolves around discrepancies between your development environment and Heroku's execution environment, specifically relating to gem dependencies and their visibility. Let’s break it down and talk solutions, using experience from past projects as guidance.

From my past work, I recall a specific project where we were migrating a fairly intricate Rails application. Locally, everything purred like a well-oiled machine, but upon deployment to Heroku, we were greeted with `ModuleNotFoundError`. After several hours, we narrowed it down to an issue with a gem that we had added relatively late in the development cycle. The problem was, although it was specified in the `Gemfile`, a particular subdirectory within its codebase wasn't being automatically recognized as a load path on Heroku. This is common: the autoloader on Heroku doesn't always pick up everything automatically. Let's get more specific about how to fix this.

First, the fundamental problem: Heroku uses a different file system and execution context compared to your local machine. The most common cause is that the gems required by your application aren't properly installed and made available within Heroku’s environment. The `ModuleNotFoundError` indicates a Ruby require or autoload call is failing to locate a required module, which could be a Ruby file or a gem. Here's where we have to be careful about how we approach the problem; the solution typically involves ensuring that your `Gemfile` and `Gemfile.lock` are accurate and complete, and that your application is loading the gem's dependencies correctly.

The initial step in troubleshooting is always the same. Examine your `Gemfile` and `Gemfile.lock` to ensure all dependencies are listed correctly and their versions are compatible with your project. A common mistake, especially in team settings, is having dependencies in the `Gemfile` that aren't fully captured by the `Gemfile.lock`. This happens when developers add gems but do not run `bundle install` locally, and then push their changes. You need to make certain that your `Gemfile.lock` reflects the exact state of your dependencies when building. To be sure, before any push, it’s advisable to execute `bundle update` followed by `bundle install` to refresh your lockfile to account for any latest changes, committing that update as well.

Beyond the `Gemfile`, a common secondary issue involves gems that include subdirectories that need to be added to the load path manually. This is frequently the case with gems that employ custom folder structures or include modules you use directly in the project. For this, you may need to adjust the Ruby load path in your Rails application to explicitly include those directories. To be clear, the load path is a setting in Ruby (represented by the `$LOAD_PATH` or `$:`) that indicates where Ruby should look for code when you use `require` or `autoload`.

Let’s see this in practical action using code examples. Assume a gem, `special_gem`, contains the subdirectory `lib/special_gem/utils/`. Your application relies on a class in `lib/special_gem/utils/helpers.rb`.

**Example 1: Adding a subdirectory to the load path within an initializer**

This example shows how to extend Rails's load paths. Create a new initializer named, say, `01_load_paths.rb` in the `config/initializers` directory:

```ruby
# config/initializers/01_load_paths.rb
require 'bundler'
Bundler.require(:default)

# Check the gems and adjust load path
Gem.loaded_specs.each do |name, spec|
  if name == 'special_gem'
     $LOAD_PATH << spec.full_gem_path + '/lib'
     $LOAD_PATH << spec.full_gem_path + '/lib/special_gem'
     $LOAD_PATH << spec.full_gem_path + '/lib/special_gem/utils'
  end
end
```

This snippet explicitly adds the subdirectories within the `special_gem` to the `$LOAD_PATH`. It’s a bit verbose, but very clear about what's happening. We can streamline this further if you want to handle loading more than one gem.

**Example 2: Streamlined Load Path Handling**

If you have a series of such gems, you may wish to extract this into a more general solution, which we can do like so:

```ruby
# config/initializers/02_load_paths_generic.rb
require 'bundler'
Bundler.require(:default)

def add_gem_load_path(gem_name, paths)
  Gem.loaded_specs.each do |name, spec|
    if name == gem_name
      paths.each do |path|
        $LOAD_PATH << spec.full_gem_path + path
      end
    end
  end
end

add_gem_load_path('special_gem', ['/lib', '/lib/special_gem', '/lib/special_gem/utils'])
add_gem_load_path('another_gem', ['/src']) # Adjust paths as needed
```

Here, the `add_gem_load_path` function provides a clear, reusable abstraction that makes modifying load paths significantly less verbose. You pass the gem’s name and then an array of the subdirectories to include. This is particularly useful as your project grows and has further dependency needs.

Another common scenario is a gem that includes its components in the `ext` directory and not `lib`. This often arises with native extensions. In these scenarios, you may also need to add the `ext` directory to the load path.

**Example 3: Adding an `ext` directory**

Suppose `native_gem` contains extensions within the `ext` directory:

```ruby
# config/initializers/03_native_ext_load_path.rb
require 'bundler'
Bundler.require(:default)

Gem.loaded_specs.each do |name, spec|
  if name == 'native_gem'
    $LOAD_PATH << spec.full_gem_path + '/ext'
    $LOAD_PATH << spec.full_gem_path + '/lib' # Standard path if needed
  end
end
```
The examples show you how to explicitly adjust your Rails app’s load paths. In a more complex project, using a custom configuration would be more appropriate than an initializer. The examples provided here are good starting points.

Beyond the load path, always ensure your buildpacks on Heroku are correctly configured, that the Ruby version is correct in both your `Gemfile` and the Heroku environment, and any environmental variables needed are also correctly configured on Heroku. It’s also a worthwhile practice to review the Heroku build logs in detail, paying close attention to any reported warnings or errors during gem installation. These logs provide invaluable information for diagnosing issues that might otherwise appear obscure.

For further reading on gem dependencies and load paths, I strongly recommend checking out "The Ruby Programming Language" by David Flanagan and Yukihiro Matsumoto for a detailed look at Ruby's fundamental principles. For specific insight on Rails autoloader and best practices, reviewing the official Rails documentation regarding its configuration options is a must. Understanding how Bundler works is key for managing your gem dependencies, and the official bundler website has extensive documentation. In general, the more you understand the underpinnings of Ruby, Bundler, and Rails's autoloading mechanisms, the easier such issues become to diagnose and resolve. These are not quick fixes, but core understanding in this area will be invaluable as your development experience grows.

I’ve found, in the end, that methodical troubleshooting, precise configurations, and a thorough understanding of the underlying system mechanics almost always lead to a resolution. It's rarely a single silver bullet, but a combination of these techniques that ultimately resolves those frustrating `ModuleNotFoundError` messages. Take your time, break the problem down, and you’ll be on your way to a successful deployment.
