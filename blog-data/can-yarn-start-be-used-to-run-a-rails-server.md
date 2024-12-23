---
title: "Can `yarn start` be used to run a Rails server?"
date: "2024-12-23"
id: "can-yarn-start-be-used-to-run-a-rails-server"
---

Okay, let's unpack this. The short answer is: not directly, and certainly not in the way you'd typically use `yarn start` with a frontend framework like React or Vue. But there's a bit more nuance to it than just a flat 'no'. I've personally encountered situations where teams tried (and failed spectacularly) to shoehorn `yarn start` into their Rails workflows, so I can speak from a place of, shall we say, hard-won experience.

The core issue stems from the fundamental differences in what `yarn` and `rails` are designed to do. `yarn` is primarily a package manager for javascript, focusing on installing, managing, and scripting tasks related to node.js applications. Its `start` command, by convention, usually invokes a development server for a single-page application (spa), often using something like webpack-dev-server or vite. Rails, on the other hand, is a full-stack web framework, built using Ruby, and it has its own robust ecosystem for managing development servers.

When we think about starting a Rails application, we usually invoke the `rails server` command or, more recently, `bin/rails server` when using a modern rails setup. This command orchestrates not just serving static assets, but also booting the entire Rails environment, including database connections, application logic, and routing. Trying to force `yarn start` to do this would be akin to using a screwdriver to hammer a nail – both are tools, but they're designed for very different purposes.

Let’s imagine a scenario I dealt with a few years back. A team decided that since their frontend was heavily reliant on npm and yarn, they’d try and streamline the start process. They configured their `package.json` to have a script: `"start": "rails server"`… This "worked" in that the server came up, but it bypassed so much of what `rails server` does under the hood. Crucially, it completely missed all the precompilation steps for the asset pipeline which means they were staring at a site with missing stylesheets and broken javascript. It was quite the mess to untangle, and demonstrated exactly why `yarn` and `rails` have their own, separate startup mechanics.

Now, that’s not to say that `yarn` has no place in a Rails project. Quite the opposite! It’s extremely valuable for managing your frontend dependencies within Rails, especially when using frameworks like React, Vue, or Angular. You'll use `yarn` to handle the dependencies for that specific part of the app.

Let's look at a concrete example. Consider a typical rails application, where you have installed react via the rails `jsbundling-rails` gem, along with the `react-rails` gem.

Here’s how your `package.json` might look:

```json
{
  "name": "my-rails-app",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@babel/preset-react": "^7.23.3",
    "esbuild": "^0.19.11"
  },
    "scripts": {
        "build": "esbuild app/javascript/*.* --bundle --outdir=app/assets/builds",
        "watch": "esbuild app/javascript/*.* --bundle --outdir=app/assets/builds --watch"
    }
}
```
And the relevant section of `config/initializers/assets.rb` would include:

```ruby
Rails.application.config.assets.paths << Rails.root.join("app", "assets", "builds")
```

Notice there isn't a `start` script here, but a `build` and `watch` script. We utilize `esbuild` here for speed (although webpack is also very commonly used in rails projects). These scripts are used to bundle all your react code. You would not use `yarn start` here to actually launch the Rails app server.

To illustrate further, let's say you wanted to trigger the javascript build within your rails application build pipeline. You could accomplish that via a rails task, for example in `lib/tasks/javascript.rake`:

```ruby
namespace :javascript do
  desc "Build javascript with yarn"
  task :build do
    sh "yarn build"
  end

  desc "Build javascript with yarn and watch for changes"
  task :watch do
    sh "yarn watch"
  end
end

```
Then you could run `rake javascript:build` or `rake javascript:watch` to compile your assets from rails.

Finally, suppose we *did* want to have some form of "start" script to automate the process of launching both our rails server *and* our javascript compilation. We could do something like this in our `package.json`:

```json
{
  "name": "my-rails-app",
  "version": "1.0.0",
   "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@babel/preset-react": "^7.23.3",
    "esbuild": "^0.19.11"
  },
  "scripts": {
        "build": "esbuild app/javascript/*.* --bundle --outdir=app/assets/builds",
        "watch": "esbuild app/javascript/*.* --bundle --outdir=app/assets/builds --watch",
         "start": "concurrently \"bin/rails server\" \"yarn watch\""
     }
}
```

Here, we're using the `concurrently` package (which you’d need to install with `yarn add concurrently`) to run two commands concurrently: `bin/rails server`, which starts the Rails server, and `yarn watch`, which starts the javascript watcher. This does not replace the rails `server` command but complements it, allowing you to more conveniently start all your dev tooling from one place.

In practice, the actual "start" script I've seen vary. Some include commands to clear caches, start specific services, or even run automated tests. This is where flexibility becomes important, and the `scripts` section of `package.json` is indeed very powerful.

However, it's critical to keep the separation of concerns clear. The `yarn start` command in these more elaborate examples is not *replacing* `bin/rails server`, but rather *augmenting* it. It's coordinating related tasks within our development environment, not attempting to replicate the core functions of the Rails server.

For a deeper understanding of the Rails asset pipeline, I'd strongly recommend reading the "Rails Asset Pipeline" guide from the official Ruby on Rails documentation itself. It’s comprehensive and will give you a solid understanding of how rails manages and compiles assets. In terms of understanding node.js package management, a thorough read-through of the yarn documentation is essential. For a general understanding of frontend build tools I would advise looking into the documentation of webpack, vite, or esbuild directly depending on what you are using in your project. Understanding the underlying mechanics allows you to work more efficiently and debug issues effectively.

In summary, while you *can* configure your `package.json` to run `rails server` under a `start` script, it’s generally not best practice, and misses the full suite of the rails server processes. `yarn` should be viewed as a tool for handling your JavaScript dependencies, and related tooling like bundling assets, not for managing a Ruby-based webserver. Use `yarn` to manage your frontend needs and the Rails CLI for backend and application related tasks. Using the `start` script as a coordinator for development processes is fine, but ensure the separation of responsibility and function remains clear.
