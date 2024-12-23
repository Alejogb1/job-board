---
title: "How can Ruby on Rails applications utilize Parcel 2?"
date: "2024-12-23"
id: "how-can-ruby-on-rails-applications-utilize-parcel-2"
---

Okay, let's dive into this. I've spent a fair amount of time over the years grappling with the front-end asset pipeline in Rails, and I've certainly seen its evolution. Integrating a system like Parcel 2 isn't just about swapping out a gem; it’s about understanding the implications of a more modern bundler within a framework primarily designed around a different approach.

From my experiences, early Rails projects often felt heavily coupled to the default asset pipeline, Sprockets. Transitioning to something like webpack was a hurdle at first, requiring a substantial refactor and a good grasp of its configurations. Moving to Parcel 2, however, presented a different set of considerations, largely centered around ease of setup and faster build times. Parcel 2's zero-configuration promise is alluring, but applying it in the context of a mature Rails project demands a specific strategy.

Let's consider the core question: how do we actually get a Rails application to leverage Parcel 2 for managing our front-end assets? We're talking about replacing or, more accurately, supplementing the standard asset pipeline to handle JavaScript, CSS, and even potentially images. It is also important to note that, from a performance perspective, opting for a modern bundler like Parcel 2 is almost always a win, particularly on large projects where the compile time overhead of the default pipeline can significantly impact developer velocity.

The critical aspect is understanding that Rails still expects assets to live in its `app/assets` directory (or some similarly configured location). Parcel 2, on the other hand, works with an entirely different philosophy, often placing source files in a separate location. This means we must establish a method to bridge the gap between where we write our code and where Rails expects to find its assets. My approach usually involves setting up a distinct directory specifically for Parcel, often something like `frontend/`.

Here's how I generally tackle this. First, you'll need Node.js installed and initialized in your Rails project:

```bash
# In the root directory of your Rails application
npm init -y
npm install parcel
```

Now, for the configuration part. Unlike webpack, Parcel's strength lies in its convention over configuration. However, we still need to tell Parcel where to look for our input and where to put the output for Rails to find it. We'll use npm scripts within `package.json` to achieve this. A basic configuration would look something like this in `package.json`:

```json
{
  "scripts": {
    "build": "parcel build frontend/index.js --dist-dir app/assets/build",
    "dev": "parcel frontend/index.js --dist-dir app/assets/build --no-cache"
  },
  "dependencies": {
    "parcel": "^2.11.0"
  },
   "browserslist": [
        "defaults"
      ]
}
```

In this example, `frontend/index.js` will be our entry point. This file is typically minimal, importing the core modules of our front-end application. Parcel 2 will then analyze this file and bundle its dependencies.

The key elements here are the `--dist-dir app/assets/build` flag, which directs the output of Parcel to a subfolder in `app/assets`, and the `--no-cache` flag for development, which will ensure changes are seen. The `browserslist` setting ensures that Parcel outputs code that is compatible with modern browsers.

Next, we’ll need to tell Rails to include these bundled assets by adding the following in `config/initializers/assets.rb`:

```ruby
Rails.application.config.assets.paths << Rails.root.join("app", "assets", "build").to_s
```

This ensures Rails recognizes the newly created directory. Finally, to actually include these assets in our layouts, in a file such as `app/views/layouts/application.html.erb`, we can use the `stylesheet_link_tag` and `javascript_include_tag` helper functions:

```html
<!DOCTYPE html>
<html>
  <head>
    <title>My Rails App</title>
    <%= csrf_meta_tags %>
    <%= csp_meta_tag %>
    <%= stylesheet_link_tag "style.css", media: "all" %>
  </head>

  <body>
    <%= yield %>
     <%= javascript_include_tag "index.js" %>
  </body>
</html>
```

Where `style.css` and `index.js` are present in `app/assets/build/`.

To complete this basic setup, let’s see this in action with a simple example. First, create the file `frontend/index.js`:

```javascript
import "./styles.css";
console.log("Parcel works!");
```

Then, create `frontend/styles.css`:

```css
body {
    background-color: beige;
    font-family: sans-serif;
}
```

After running `npm run build`, you will see `style.css` and `index.js` in `app/assets/build/`.
Now, if you run `rails s` and open the page, you’ll see that `Parcel works!` is logged in the browser console, the background is beige and the page uses a sans-serif font.

While this setup works, it's worthwhile considering more advanced cases. For instance, if you are working on a project with various entry points, you may want to consider using a more advanced configuration. Consider a scenario where we have two entry points: `frontend/admin.js` and `frontend/customer.js`. We can adapt our `package.json` configuration:

```json
{
  "scripts": {
    "build": "parcel build frontend/admin.js frontend/customer.js --dist-dir app/assets/build",
    "dev": "parcel frontend/admin.js frontend/customer.js --dist-dir app/assets/build --no-cache"
  },
  "dependencies": {
    "parcel": "^2.11.0"
  },
 "browserslist": [
        "defaults"
      ]
}
```

In this case, both `admin.js` and `customer.js` will be bundled separately. This will create `admin.js` and `customer.js`, plus their respective css files (if they import css files), in `app/assets/build`. You would then include them using separate `javascript_include_tag` in the corresponding views.

This approach separates concerns more clearly, which is generally recommended as it will result in less code needing to be loaded on any given page.
While Parcel excels at simplicity, for complex setups, I also strongly advise familiarizing yourself with the official Parcel documentation. Also, regarding general front-end bundling, "JavaScript Application Design" by Nicolas Bevacqua is a very informative resource that outlines different approaches to frontend code organization, which can help when designing how your various JavaScript bundles will be structured. "High Performance Browser Networking" by Ilya Grigorik offers insights into how browsers load resources and can inform your bundling strategy for optimal performance.

In my experience, using Parcel 2 in Rails involves understanding the core difference in philosophy of the asset management tools and creating a seamless integration bridge. It requires a small amount of manual setup, but the improvements in build times and ease of configuration more than compensate. The examples provided should offer a solid starting point for those looking to make the jump. As with any technology, understanding the underlying principles helps to navigate any hurdles that might arise. The techniques above should cover most typical use-cases, but always be prepared to adjust and tweak for your specific project needs.
