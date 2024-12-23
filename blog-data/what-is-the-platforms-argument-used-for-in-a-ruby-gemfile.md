---
title: "What is the 'platforms' argument used for in a Ruby Gemfile?"
date: "2024-12-23"
id: "what-is-the-platforms-argument-used-for-in-a-ruby-gemfile"
---

Okay, let's talk about `platforms` in a Ruby `Gemfile`. I’ve seen it trip up newcomers and even seasoned developers a few times, especially when dealing with cross-platform deployments. It's a seemingly simple directive, but it packs a lot of power in terms of dependency management and ensuring gem compatibility across different operating systems and architectures.

It might seem trivial at first glance, but the `platforms` argument isn't there just to add complexity. It's about specificity. Imagine working on a project a few years back; we had a complex image processing pipeline that was initially developed on macOS. When we started deploying to our production linux servers, we ran into all kinds of gem compatibility issues. Certain native extensions within specific gems wouldn't compile or function correctly on linux. That’s where using `platforms` became essential. We were forced to rethink how we were managing dependencies based on our target environments.

Essentially, the `platforms` argument in a `Gemfile` lets you specify gems that should only be installed on particular platforms. A platform, in this context, can refer to a combination of operating system, processor architecture, and even the Ruby implementation. This is crucial when your project has different requirements depending on the deployment environment. Without it, your bundle might attempt to install gems that simply won't work, leading to build errors, runtime exceptions, or worse, silent failures.

A good example is when you’re using a gem like `nokogiri`, which relies heavily on compiled extensions. The macOS versions of the gem's native libraries differ from their linux counterparts. If your Gemfile doesn't account for this, a bundle install on the wrong platform will likely break your builds. In my experience, ignoring this issue leads to frustrating debugging sessions, and usually involves spending valuable time tracking down library incompatibility issues instead of developing actual product features.

Let’s consider a few real world use cases and corresponding code examples.

**Example 1: Different Database Drivers**

Let's say you're developing locally on a macOS machine, using a sqlite database for convenience, but your production server uses postgresql. Here’s how you'd handle that.

```ruby
source 'https://rubygems.org'

gem 'rails', '~> 7.0'
gem 'sqlite3' , platforms: :development # sqlite3 only needed during development

group :production do
 gem 'pg' # postgres driver for production
end

```

In this setup, when you run `bundle install` on your local machine which has the `:development` environment, you get the `sqlite3` gem. On the other hand, when you bundle install on your production machine with `bundle install --without development`, you only get the production-specific `pg` gem. I would recommend keeping your development environment clean by keeping all non-production dependencies within that platform group.

**Example 2: OS-Specific Native Extensions**

Here's a scenario more closely related to the `nokogiri` example we mentioned. Let's say you need to use a particular gem with native extensions optimized differently for macOS and linux:

```ruby
source 'https://rubygems.org'

gem 'rails', '~> 7.0'

gem 'os_specific_gem', platforms: :x86_64_darwin # for macOS x86_64
gem 'os_specific_gem', platforms: :x86_64_linux  # for linux x86_64
gem 'os_specific_gem', platforms: :arm64_darwin #for macOS ARM64
gem 'os_specific_gem', platforms: :arm64_linux # for Linux ARM64


```

As you see, rather than just using `:linux` and `:darwin`, which can be too broad, you are specifying the architecture as well, `x86_64` and `arm64`. This ensures that the correct binary for each OS and architecture combination is installed. This can be a lifesaver when you start working on arm64 architectures, or deploy cross arch platforms.

**Example 3: Specific Ruby Implementation**

Finally, let’s consider a scenario where your gem needs a particular Ruby implementation. Some gems may rely on the features of a specific Ruby implementation, such as JRuby or TruffleRuby.

```ruby
source 'https://rubygems.org'

gem 'rails', '~> 7.0'
gem 'truffleruby_specific_gem', platforms: :truffleruby

```

In this situation, the `truffleruby_specific_gem` will only be included in your bundle if the Ruby implementation is TruffleRuby. This avoids any clashes or issues that could arise from installing the gem in a standard MRI ruby setup. This specific gem is going to be only used in the special implementation and it's not going to cause any dependency issues in the common scenarios.

In practice, I tend to make use of the `:development`, `:test`, and `:production` groups and only very rarely delve into the specifics of archicture or ruby implementation. In most cases, the `:production` group provides the perfect way to isolate any problematic or platform specific dependencies.

The key takeaway here is that understanding the `platforms` argument is critical for managing dependencies in a cross-platform or complex environment. Ignoring it can lead to significant headaches down the line, especially in production where unexpected issues can impact end-users.

To further understand the complexities and nuances of gem dependencies I highly recommend exploring “Understanding Bundler” by Aaron Patterson and "Ruby Under a Microscope" by Pat Shaughnessy. "The Pragmatic Programmer" by Andrew Hunt and David Thomas is another great overall resource, offering best practices on project management and how to think about potential issues beforehand. While these books don’t focus directly on bundler's platforms argument, they provide a solid foundation for understanding why the issues that `platforms` aims to solve are important. Understanding your dependencies and how they fit within your broader architecture is paramount and reading these books will give you a solid advantage.

Using the `platforms` keyword with a clear understanding of your environment can significantly improve the reliability and portability of your Ruby applications.
