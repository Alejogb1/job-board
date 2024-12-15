---
title: "Why isn't SimpleCov processing statistics?"
date: "2024-12-15"
id: "why-isnt-simplecov-processing-statistics"
---

alright, so you're having trouble with simplecov not generating coverage stats, huh? i've been there, trust me. it's one of those things that can make you question your entire setup. let's unpack this. it’s not always immediately obvious, and there are several typical culprits at play.

first off, let's be sure about the basics. simplecov, at its core, is a code coverage tool for ruby. it works by hooking into the ruby runtime and tracking which lines of your code are executed during your test runs. if it's not producing stats, it means this instrumentation isn't happening or the data isn't making it to simplecov.

i've seen this happen a lot, and usually, it boils down to one of a few key issues. i remember one time, back when i was working on a legacy rails app (you know, the kind with the ancient codebase that’s like archaeological artifact), i spent almost a whole day on this. it turned out i was loading simplecov in the wrong place. rookie mistake, but we all do it.

so, let’s start with the setup. simplecov needs to be required *before* your application code and your test framework. this is absolutely crucial. it has to be the first thing that fires up in your test suite. if it’s loaded after your tests start running, it simply won’t instrument the code execution. it will miss those crucial details.

here’s how a typical setup should look in your `test/test_helper.rb` file, or something similar if you’re using another test framework like rspec:

```ruby
require 'simplecov'
SimpleCov.start do
  add_filter "/test/" # optional, filters out test files
  add_filter "/config/" # optional, filters out config files
end

ENV['RAILS_ENV'] ||= 'test'
require_relative '../config/environment'
require 'rails/test_help'
```

that little snippet above is a lifesaver. you'll notice that `require 'simplecov'` and `SimpleCov.start` are at the very top, before loading your rails environment or anything else. this order matters immensely. if this isn't the case, move it up and try your tests again. in my experience, most of the time this sorts out the problem.

another common mistake is related to where and how you’re running your tests. simplecov relies on a process's context, specifically the one where the tests are executed, to track code execution. if you’re running tests in a subprocess or through some sort of parallel testing setup, without correctly configuring simplecov to work with that structure, the code coverage data is going to get lost in transit. it’s like trying to send mail through the wind. it’s not going to arrive.

i had this happen once when i started experimenting with parallel tests to speed up things. i was using parallel_tests gem. what i had to do was setup a bit of code to merge the coverage data from those multiple parallel test processes.

here is a code snippet that i used for that case:

```ruby
if ENV['PARALLEL_TEST_GROUPS']
  require 'simplecov'
  SimpleCov.command_name "test:#{ENV['TEST_ENV_NUMBER']}"
  SimpleCov.start
else
  require 'simplecov'
  SimpleCov.start do
    add_filter "/test/"
    add_filter "/config/"
  end
end

ENV['RAILS_ENV'] ||= 'test'
require_relative '../config/environment'
require 'rails/test_help'
```

this checks if parallel tests are being run and then sets `command_name` accordingly. the important part is that when you’re not using parallel tests it falls back to the original setup. this way it will avoid any conflicts and make simplecov function as usual, regardless if tests are run in parallel.

furthermore, also check that your test framework isn’t interfering in some way with simplecov. some test frameworks are very opinionated, and you may need to configure simplecov to work alongside them. while rare, sometimes there are odd interactions that need a bit more specific configuration in order for things to operate smoothly. it's unlikely, but worth mentioning. it's a good habit to check any framework docs, just in case it is mentioned there.

also verify if you've added any file filters. sometimes, people unknowingly add filters that inadvertently exclude the very files they expect coverage from. double check your `SimpleCov.start` block. maybe you've got a filter that is too broad. i've done this myself, accidentally filtering out a bunch of folders, wondering why some files were being ignored. it's like when you're trying to fix a leaky faucet, and you end up turning off the water to the entire house, happens to the best of us.

here is an example of a too-broad filter, you want to be careful:

```ruby
SimpleCov.start do
  add_filter "/app/" # <- this is too broad
end
```

if you include `/app/`, simplecov will filter out the whole app folder, leaving no files to track for coverage. it needs to be specific, like this:

```ruby
SimpleCov.start do
  add_filter "/test/"
  add_filter "/config/"
  add_filter "/spec/"
end
```

finally, and this one seems odd but has happened, make sure your simplecov gem itself is correctly installed and has no version incompatibilities with your ruby version or other gems. weird version issues can manifest in very surprising ways. it's always good to check your gemfile.lock file or reinstall simplecov if anything seems odd. gem issues are no joke, can mess with your mind.

as for further resources, i suggest checking out the simplecov gem's official documentation, it’s usually the best place to start. also, the "working effectively with unit tests" book by maurice naftalin and matt newkirk has a detailed chapter on code coverage tools and best practices. and finally, if you want to better understand how code coverage works, the "art of software testing" by glenford myers is a solid resource. it's a bit theoretical but it's a good read for understanding the principles.

it may seem like a lot, but usually the solution is within one of these areas. double-check your setup, placement of simplecov, filters and parallel tests. most of the times, it is that simple. let me know if you need anything else.
