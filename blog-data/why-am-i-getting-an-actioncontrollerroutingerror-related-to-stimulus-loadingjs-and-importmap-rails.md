---
title: "Why am I getting an `ActionController::RoutingError` related to `stimulus-loading.js` and `importmap-rails`?"
date: "2024-12-23"
id: "why-am-i-getting-an-actioncontrollerroutingerror-related-to-stimulus-loadingjs-and-importmap-rails"
---

,  I've seen this exact scenario crop up a few times in projects, and it can be a bit frustrating if you’re not familiar with the intricacies of how `importmap-rails` and Stimulus interact. It usually stems from a misalignment in how your application expects to find the necessary JavaScript modules, and here’s a breakdown of why and what you can do.

The `ActionController::RoutingError` in this case isn't directly about your Rails routes in the traditional sense, but rather, it indicates that the browser is requesting a resource (in this case, `stimulus-loading.js`) at a URL that Rails can't map to a file on disk. This happens because `importmap-rails` alters how your JavaScript dependencies are handled. Instead of relying on Webpack or a similar bundler, it leverages browser-native JavaScript modules and an import map.

Essentially, you've got the following sequence: the browser (likely via Stimulus) tries to load `stimulus-loading.js` as a module, the import map attempts to resolve it, and, if no mapping is found or it's not served correctly as a static asset, a routing error will occur on the Rails side due to no matching route.

This can arise from a few key areas:

1.  **Missing Mapping:** The most common culprit is an incomplete or incorrect import map. The `importmap-rails` gem maintains a JSON file (typically `config/importmap.rb` or similar) that specifies how JavaScript module names (like `stimulus-loading`) translate to URLs. If this mapping is missing or incorrect, the browser won’t know where to fetch the resource.

2.  **Incorrect or Absent Assets:** Even if the mapping is correct, the actual JavaScript file might not be available at the location specified. This could be because the file wasn't precompiled, or if you're experimenting with development setup, not properly served as a static asset.

3. **Import statement not being used with importmap:** `importmap-rails` does not magically make any javascript available, if the modules are not defined in the `importmap.rb` or imported in the `application.js` file then it won't be available

Let me illustrate with some practical examples.

**Example 1: Missing Mapping**

Imagine your `config/importmap.rb` looks like this:

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "https://ga.jspm.io/npm:@hotwired/stimulus@3.2.2/stimulus.js"
```

Notice there's no `stimulus-loading` defined. This would result in the `ActionController::RoutingError`. To resolve this, you'd need to add:

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "https://ga.jspm.io/npm:@hotwired/stimulus@3.2.2/stimulus.js"
pin "@hotwired/stimulus-loading", to: "https://ga.jspm.io/npm:@hotwired/stimulus-loading@2.0.0/index.js"
```

The `pin` command tells `importmap-rails` that whenever the browser requests `@hotwired/stimulus-loading`, it should fetch the resource from the specified URL. Now when the browser tries to load it, it should know the exact URL to call, and the `ActionController::RoutingError` will disappear. I’ve personally seen this happen more than once where I’d forgotten the stimulus-loading piece.

**Example 2: Incorrect Assets Handling**

In development, Rails usually serves assets from the `public` directory. However, If your import map points to a location within your javascript folder you need to check if the assets are being served correctly. To demonstrate, let’s say your `importmap.rb` contains:

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "https://ga.jspm.io/npm:@hotwired/stimulus@3.2.2/stimulus.js"
pin "@hotwired/stimulus-loading", to: "app/javascript/stimulus_loading.js"
```

And you intend to use a local copy of `stimulus-loading.js` for customization in development, assuming that it exists in `app/javascript/stimulus_loading.js`. This means it isn’t going to be a static asset and rails won’t know how to route to this file, instead it needs to be imported in your application.js with the correct location. In this case you will still have an `ActionController::RoutingError`.

```javascript
// app/javascript/application.js
import "./stimulus_loading";
```

In a production or testing environment, make sure the precompiled assets are correctly served by your web server.

**Example 3: Incorrect Import statements.**

Even if you have correctly installed and added `@hotwired/stimulus-loading` to `importmap.rb`, you still need to import it into your `application.js` file.

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "https://ga.jspm.io/npm:@hotwired/stimulus@3.2.2/stimulus.js"
pin "@hotwired/stimulus-loading", to: "https://ga.jspm.io/npm:@hotwired/stimulus-loading@2.0.0/index.js"
```

You might think that it would now be accessible in your application, but unless you add the following line into your `application.js` you won't have the module available in your app.

```javascript
import "@hotwired/stimulus-loading"
```

In most applications you do not need to add this line as the module is loaded automatically when loading a Stimulus controller, but if it isn't there then it will be worth checking.

**Troubleshooting Steps:**

1.  **Inspect Your Import Map:** Double-check your `config/importmap.rb` or equivalent. Make sure `@hotwired/stimulus-loading` (or any other failing module) is pinned, with the correct version. A common mistake is a version mismatch between what's specified in the import map and what is actually in the `package.json`, which although not used directly by `importmap-rails` can sometimes be useful to help ensure that you are using a correct version.
2.  **Verify Asset URLs:** Check the exact URLs specified in your import map to make sure they’re valid and accessible. It is recommended to use something like jsdelivr or esm.sh to ensure a stable CDN. Be careful when using `jspm.io` as it can be difficult to identify the actual URL being pulled and when looking at versions it can be ambiguous, as in the case above I am using `@hotwired/stimulus-loading@2.0.0` however the `index.js` file is located at `https://ga.jspm.io/npm:@hotwired/stimulus-loading@2.0.0/index.js`, and other versions are available, meaning you can accidentally use a different version without knowing it.
3.  **Precompile Assets:** If you are using your own `stimulus-loading.js` file, ensure your application is precompiling the assets correctly. Check your web server's configuration for how it serves static files.
4.  **Restart your server:** if you add a new dependency or change your importmap it is essential that you restart your rails application to pickup the changes.
5.  **Check the browser console:** Often times the browser console can point directly to the resource which is causing the issue, which can be invaluable for tracking down the specific problem

**Recommended Resources:**

*   **The `importmap-rails` gem documentation:** The official documentation provides the most detailed explanation of how to use it and its features. Focus especially on the configuration sections and asset loading in different environments.
*   **Hotwired Documentation:** As `@hotwired/stimulus` and `stimulus-loading` are part of the Hotwired ecosystem, referencing the official documentation will help clarify how they work together, specifically if you run into trouble with another module.
*   **"JavaScript Everywhere" by Adam D. Scott:** While not exclusively focused on Rails, this book provides a comprehensive understanding of modern JavaScript module systems, which can be beneficial in understanding the reasoning behind `importmap-rails`.
*   **"Rails 7: From Zero to Deploy" by Andrea Leopardi:** This book has a great section covering the specifics of using importmap-rails in Rails, and could be very helpful for setting it up correctly.

These steps and resources should provide a solid foundation for resolving the `ActionController::RoutingError` in your situation. Remember that these tools are designed to streamline development when used correctly and a small misunderstanding can cause disproportionate problems.
