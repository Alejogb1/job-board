---
title: "How can I pin npm/yarn packages in Rails 7 using importmap?"
date: "2024-12-23"
id: "how-can-i-pin-npmyarn-packages-in-rails-7-using-importmap"
---

Alright, let's talk about pinning npm/yarn packages with importmap in Rails 7. It's a problem I tackled a fair bit back in my days maintaining a fairly large SaaS application using Rails, and it's definitely an area that can cause confusion if not approached methodically. The core issue stems from the way importmaps are designed: they map module specifiers directly to URLs, meaning we’re bypassing the typical node module resolution process. Consequently, direct imports from node_modules simply don’t work. To bridge this gap, we have to explicitly tell importmap how to find and load our needed npm/yarn packages.

When you're looking to integrate npm packages into a Rails 7 app using importmap, remember that importmap itself doesn’t install or manage dependencies in the conventional manner that npm or yarn does. It relies on pre-built package files, which are typically obtained via a build tool. The process involves several key steps: you have your node dependency management (npm or yarn), which gets you the package, a bundling step (typically using esbuild, webpack, or a similar bundler) to create a deployable javascript file or files, and finally the use of importmap within Rails to map your module names to the URLs where those files are served.

Here’s how I typically go about it, and it’s a method that’s served me well over multiple projects. I'll break it down into discrete steps and then follow up with concrete examples.

First, you’ll use npm or yarn to install your desired packages. This is a standard step and you should have a `package.json` and a `package-lock.json` or `yarn.lock` set up in your project's root. This ensures that package versions are correctly locked. Secondly, you need to utilize a bundling tool to bundle your code and node module dependencies into one or more javascript files. For instance, we might use esbuild, which is a lightweight and fast option for this kind of task. This bundled file (or files) is then placed into a public-accessible directory, usually `app/assets/builds`.

Following bundling, the next step is configuring your `importmap.rb` file. Here, you'll define explicit mappings between module specifiers (like `'lodash'`) and the URL to the bundled JavaScript file. The crucial aspect here is that the mapped URLs should point to the location of the bundled file produced by the build tool, taking the `/assets/` URL prefix used by rails.

Finally, and equally important, you then use the mapped import specifiers within your javascript code in rails. So, if you've mapped lodash, it should be `import _ from 'lodash'` rather than any other form.

Let’s solidify this with examples:

**Example 1: Using Lodash with esbuild**

1.  **Install Lodash:**

    ```bash
    npm install lodash
    ```

2.  **Configure esbuild:** You might use a command similar to this in your `package.json` script definitions or your build configuration:

    ```json
    "scripts": {
      "build": "esbuild app/javascript/*.* --bundle --outdir=app/assets/builds --public-path=/assets --sourcemap"
    }
    ```
    This command takes all javascript files in `app/javascript`, bundles them, outputs the result into `app/assets/builds`, and tells it to use `/assets` as a public path.
   You need to ensure you call this build script whenever you change your javascript or need to update your deployed dependencies.

3.  **Create `app/javascript/application.js` (or modify yours)**

    ```javascript
    import _ from 'lodash';

    console.log(_.shuffle([1, 2, 3, 4]));
    ```

4.  **Update `config/importmap.rb`:**
    ```ruby
    pin "application", preload: true
    pin "lodash", to: "lodash.js", preload: true
    ```

    Note that we have implicitly named the outputted bundled file from esbuild as "lodash.js". If you have multiple entrypoints or produce multiple bundled files, you will need to adjust this.

    After building you will have `app/assets/builds/application.js` and potentially other javascript files including the `lodash.js` files.

    In this scenario you are using lodash from `application.js` after having explicitly mapped it in `importmap.rb`.

**Example 2: Using a Specific Version of a Package**

  This is where the `package-lock.json` or `yarn.lock` files come in handy. Suppose you need to lock in a specific minor version of a library such as moment.js:

  1. Install moment using `npm install moment@2.29.4` or similar with yarn. This will update your `package.json` and either `package-lock.json` or `yarn.lock`. You'll be sure to be getting version 2.29.4.

  2. Your build configuration will be essentially the same as in the previous example, including the entrypoint.

  3. You'd then update your importmap to explicitly reference the bundled file that contains moment.js.
    ```ruby
    pin "moment", to: "moment.js", preload: true
    ```
  4. Within your javascript:
      ```javascript
      import moment from 'moment';
      console.log(moment().format('MMMM Do YYYY, h:mm:ss a'));
      ```
      This will ensure you are importing the correct version of moment.js from your bundled file, even if a new version is available in the npm registry.

**Example 3: External Dependencies**

 Sometimes, you might rely on external JavaScript files. These could be from a CDN. Importmap is also capable of referencing these, although it’s generally advisable to bundle your core dependencies for performance and consistency of versions. Nonetheless, the process of referencing external javascript files is relatively straightforward.

1. We assume for the sake of example that you rely on an external library called 'my-library' and this resides in a CDN at `https://cdn.example.com/my-library.js`

2. You'd then update `config/importmap.rb` to include a reference to that URL.
    ```ruby
    pin "my-library", to: "https://cdn.example.com/my-library.js", preload: true
    ```

3.  In your javascript you can then import it as:
    ```javascript
    import myLibrary from 'my-library';
    myLibrary.doSomething();
    ```

This demonstrates how you map an external dependency using importmap.

Regarding resources, I highly recommend *High Performance JavaScript* by Nicholas C. Zakas for understanding javascript performance bottlenecks. It’s a classic for a reason. For a good foundation of module bundling and build tools, the official documentation for esbuild and webpack is invaluable. Specifically, the section regarding module resolution in the webpack documentation can offer a deeper understanding. Also, for general best practices on dependency management and Javascript in Rails, the official Ruby on Rails documentation and the community guides are very useful. They tend to provide comprehensive explanations. Also, the *Eloquent Javascript* book by Marijn Haverbeke provides a robust foundation in Javascript.

In my experiences, a solid understanding of the underlying bundler (esbuild or webpack etc.) and importmap mechanics is crucial for debugging and setting up dependency management correctly. It’s not a magic bullet; it requires attention to detail, especially when dealing with nuanced versioning issues and complex dependency graphs. However, once you have the flow down, it’s quite a smooth process for managing front-end libraries in Rails. Remember that importmaps are more about connecting javascript URLs to import specifiers than traditional dependency management itself; npm or yarn (or pnpm, etc.) handles that part. The key is correctly bundling and mapping.
