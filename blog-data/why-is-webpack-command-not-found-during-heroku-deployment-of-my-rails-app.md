---
title: "Why is 'webpack' command not found during Heroku deployment of my Rails app?"
date: "2024-12-23"
id: "why-is-webpack-command-not-found-during-heroku-deployment-of-my-rails-app"
---

Alright, let's address this webpack issue. It's a classic head-scratcher, and I've bumped into this exact situation more times than I care to remember, particularly with Rails deployments on Heroku. It usually boils down to a combination of environmental discrepancies and package management hiccups. Let’s unpack this, because "webpack command not found" is rarely a webpack problem *itself*, more a symptom of something else.

The first place I always look, and where my years have taught me is most fruitful, is the Heroku build process. Heroku builds your application in an environment that isn’t always a direct mirror of your development machine. This is deliberate; it allows for a clean, consistent build regardless of the developer's setup. The problem is that this 'clean' environment often lacks dependencies that are assumed to be present locally.

Specifically, when deploying a Rails application that incorporates webpack for asset management, a few potential pitfalls arise. Primarily, Heroku's build process uses buildpacks, and the standard Ruby buildpack doesn't natively include webpack as a global command that's immediately available. In other words, your local machine likely has node.js and the `webpack-cli` installed globally or at least accessible within your project, but these are not pre-installed within the Heroku build environment unless explicitly declared. This results in Heroku attempting to execute `webpack` during the asset precompilation stage (usually part of the deploy), but finding that command does not exist.

Now, how do we rectify this? Well, there's usually more than one way to skin a cat, but I find the following approach tends to be the most straightforward. The core strategy revolves around two things: first, making sure node.js is present in the build environment; and second, ensuring that the specific `webpack` command is scoped within your project and not relying on global installations.

Let's talk through three practical code examples, and then I'll point to some good resources at the end.

**Example 1: Adding Node.js and NPM as Build Dependencies using `package.json`**

This is the most common solution, and the most dependable. We're going to explicitly state our node.js needs. This involves creating a `package.json` file in the root of your Rails application, if you don't already have one. If you've been using npm or yarn for any JavaScript dependencies, you probably have one, just be aware where it lives. Within `package.json`, you will define the node version and any required node packages for the build. Here’s how that might look:

```json
{
  "name": "your-rails-app",
  "version": "1.0.0",
  "engines": {
    "node": "16.x"
  },
  "dependencies": {
  }
  ,
  "devDependencies": {
      "webpack": "^5.0.0",
     "webpack-cli": "^4.0.0"
  }
}
```

Here, we specify the compatible node version in the `"engines"` section and add `webpack` and `webpack-cli` to the `"devDependencies"` section. This ensures Heroku's build process will install node.js, npm (or yarn if `yarn.lock` exists, Heroku detects this) and download these dependencies. Crucially, since these are part of the project's dev dependencies, `webpack` will be available locally in your `node_modules` directory and executable using npm scripts.

**Example 2: Modifying the `package.json` script for executing webpack**

Once webpack is available in `node_modules`, we need to ensure that Rails (during asset precompilation) executes it correctly. Rails often uses a command like `bin/webpack` or a similar command based on a setting that’s within the `config/webpacker.yml` file. In many standard Rails setups you will find a `rails webpacker:compile` command being used within the `assets:precompile` task. We will ensure this is properly running the webpack command inside the context of the node.js environment we have now configured. We will leverage the npm scripts capability of `package.json`. We can make a simple change, we add a script section within package.json and provide a method to execute the `webpack` command:

```json
{
  "name": "your-rails-app",
  "version": "1.0.0",
    "engines": {
        "node": "16.x"
    },
  "dependencies": {
  }
  ,
   "devDependencies": {
        "webpack": "^5.0.0",
        "webpack-cli": "^4.0.0"
    },
    "scripts": {
        "build": "webpack --mode production"
  }
}
```

Now we have added a `build` command, which we will refer to later.

**Example 3: Directing Rails to execute the npm script for webpack**

Finally, we need to make sure the Rails asset pipeline is triggered to use the script we have created. You may find this done in a number of ways. The usual practice is to modify your `rails/config/environments/production.rb` configuration file, and this method will also cover you for any Heroku deploy. We will add a setting within this configuration file to direct rails asset precompile step to use our custom npm command to execute the webpack compiler. The configuration line will look like:

```ruby
  config.assets.precompile += [
   'application.js',
   'application.css',
   'packs/application.js'
   ]
 config.webpacker.check_missing_packages = false # optional but helpful
  config.assets.initialize_on_precompile = false

   # add this line to configure Rails to execute webpack via our npm script
  config.assets.configure do |env|
    env.register_transformer 'text/jsx', 'text/jsx', proc {|input| `npm run build` }
  end
```

Note the inclusion of  `config.webpacker.check_missing_packages = false` which is not always required, but sometimes helps prevent errors. Also note the use of  `config.assets.initialize_on_precompile = false` this prevents Rails from performing unneeded database interactions during the precompilation step.

By defining `config.assets.configure`, we direct Rails precompilation stage to execute the command `npm run build`. As specified in our `package.json`, this command in turn executes the `webpack` command that is now local to our project.

By implementing these steps, Heroku should now be able to correctly build your assets during deployment and will not encounter the "webpack command not found" error. The critical factor here is not simply *having* webpack, but explicitly instructing Heroku's build process where to find it and how to execute it.

**Resources:**

For further, more in-depth exploration, I recommend these specific resources:

*   **"High Performance Browser Networking" by Ilya Grigorik**: This book (available online) provides a detailed explanation of how asset pipelines work and how to optimise asset delivery. It's useful in understanding why certain configurations are essential.

*   **"Pro Git" by Scott Chacon and Ben Straub:** Although not specifically about webpack, a thorough understanding of git and version control practices (as discussed in this book) is invaluable for understanding and debugging deployment issues, since deployment and versioning are tightly interconnected.

*   **The official Heroku documentation, specifically on buildpacks:** This is essential reading for anyone deploying to Heroku, as it details the specifics of each build process. The documentation for the Ruby buildpack, as well as the Node.js one, is highly pertinent.

*   **Webpack's official documentation**: When addressing specific webpack configurations, nothing beats the source itself. Pay particular attention to the sections about configuration, output, and module resolution.

These resources, used in conjunction with the information discussed above, should provide a strong foundation for resolving not only this particular issue but for gaining a deeper understanding of the underlying mechanisms of web application deployment. The devil is often in the details, so understanding the nuances of the entire chain, from build process to deployment, is crucial. I hope this helps steer you back on track and towards a successful deployment.
