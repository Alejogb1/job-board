---
title: "Is webpacker causing performance bottlenecks?"
date: "2024-12-23"
id: "is-webpacker-causing-performance-bottlenecks"
---

Let's get this sorted then. I've seen this particular issue pop up more times than I care to count over the years, and it's usually a matter of peeling back the layers to find the culprit, rather than assuming webpacker itself is inherently flawed. It's true, though, that a poorly configured webpacker setup can absolutely become a performance bottleneck in a Rails application, primarily due to its role in bundling assets.

Now, I remember back at my old shop, we migrated a legacy Rails app to include webpacker, thinking we'd get all the modern javascript goodness. Initially, things were rosy, until the app started to scale. Suddenly, page load times became a serious issue, especially on the javascript-heavy parts of the site. We initially assumed it was the javascript itself, but deeper investigation pointed at our webpacker configuration as the main bottleneck. It wasn't that webpacker *couldn't* perform; it was that *our usage of it* was suboptimal.

The problem, typically, isn’t webpacker's architecture per se, but the way we're telling it to operate. One common mistake is failing to leverage code splitting effectively. Instead of chunking your javascript into smaller, more manageable parts loaded only when required, everything gets bundled into a monstrous `application.js` file. This means users download all the javascript, even if they’re only accessing a small part of the website. That overhead adds up quickly. This was exactly what hit us.

Another frequent culprit is inefficient asset processing. We tend to throw in large, unoptimized images and forget to configure loaders correctly to compress and handle these properly. Similarly, development builds tend to include a lot of bloat – debugging helpers, source maps – which are essential for development but certainly not production. We learned that lesson the hard way too; deploying a development build by accident to our staging environment revealed a lot about how we were not configuring things properly for different environments.

Finally, excessive or unnecessary libraries being included can also create bottlenecks. Think of all those tiny javascript libraries you've installed, each adding its small overhead to the final bundle size. Even libraries that aren't being directly used often remain within the bundle.

To illustrate these points, consider these concrete examples:

**Example 1: The monolithic bundle (inefficient code splitting)**

Imagine we have a basic application structure that imports a few modules:

```javascript
// app/javascript/packs/application.js

import "./moduleA";
import "./moduleB";
import "./moduleC";
import "some-external-library";

// app/javascript/moduleA.js
console.log("Module A loaded");

// app/javascript/moduleB.js
console.log("Module B loaded");

// app/javascript/moduleC.js
console.log("Module C loaded");
```

If you run webpacker without any specific splitting instructions, this will get bundled into a single file, `application.js`. Even if a user only visits a page requiring moduleA, all of the code for `moduleB`, `moduleC` and `some-external-library` will also be downloaded. Now let’s see what happens when we implement code splitting.

**Example 2: Implementing code splitting with dynamic imports**

Here's how we can modify the application to load moduleB and moduleC on demand:

```javascript
// app/javascript/packs/application.js
import "./moduleA";

document.addEventListener('DOMContentLoaded', () => {
  if(document.getElementById("some-element-that-needs-module-b")){
      import(/* webpackChunkName: "moduleB" */ "./moduleB").then(module => {
          console.log("Module B Loaded dynamically:", module);
      });
  }

  if(document.getElementById("some-element-that-needs-module-c")) {
        import(/* webpackChunkName: "moduleC" */ "./moduleC").then(module =>{
           console.log("Module C Loaded dynamically:", module);
        });
    }
});

// app/javascript/moduleA.js
console.log("Module A loaded");

// app/javascript/moduleB.js
console.log("Module B loaded");

// app/javascript/moduleC.js
console.log("Module C loaded");

```

By using dynamic `import()` statements, along with chunk names via the comment, we're telling webpacker to create separate bundles for moduleB and moduleC. These chunks are only loaded when the corresponding section of the page is viewed (or when an event is triggered that needs to access it), thereby significantly reducing the initial page load time. This is a fundamental aspect of enhancing performance, and it was a major change we had to implement for our past issues.

**Example 3: Setting appropriate compression and optimization settings**

The following demonstrates how to adjust webpacker's configuration file to optimize the assets:

```javascript
// config/webpack/production.js
const { environment } = require('@rails/webpacker')
const TerserPlugin = require('terser-webpack-plugin')
const CompressionPlugin = require("compression-webpack-plugin");


environment.optimization = {
  minimizer: [
    new TerserPlugin({
      terserOptions: {
        compress: {
          drop_console: true,
        },
      },
      extractComments: false,
    }),
  ],
};


environment.plugins.prepend(
    'CompressionPlugin',
    new CompressionPlugin({
    algorithm: 'gzip',
    test: /\.(js|css|html|svg|json)$/,
    threshold: 10240,
    minRatio: 0.8
}));


module.exports = environment
```

Here, we're adding configurations for terser (for minification) and gzip compression. This configuration will result in smaller bundles on production environments and reduce transfer times.

When looking to debug this, don't blindly blame webpacker. You should inspect the output bundles (using webpack bundle analyzer is helpful here), understand which components load when, and optimize accordingly. We learned to be particularly wary of 'black box' dependencies; those that include a lot of extra stuff and we should instead choose the lightest-weight version possible when available. Also, make sure your build process for production is set up to minify and compress assets.

To delve deeper into webpack optimization techniques, I would recommend “Surviving Webpack: An Introduction and Guide” by Sean Larkin. It provides a comprehensive view of webpack's functionalities and helps understand configuration options much more in detail. For a more broad look into performance in web applications, "High Performance Web Sites" by Steve Souders is a must-read. These resources helped us immensely in our journey to improve our webpacker setup.

In my experience, webpacker is a powerful tool that can contribute to fantastic performance when configured and used thoughtfully. It's very easy to default to just accepting the defaults and only paying attention when things go wrong, but proactive optimisation can save significant pain further down the line. The key is not to see it as a black box, but rather as a powerful engine that you need to tune and configure for optimum performance. It takes effort, but the performance gains can be immense.
