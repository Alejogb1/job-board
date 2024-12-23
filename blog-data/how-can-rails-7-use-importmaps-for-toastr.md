---
title: "How can Rails 7 use importmaps for toastr?"
date: "2024-12-23"
id: "how-can-rails-7-use-importmaps-for-toastr"
---

Alright, let's talk about getting toastr integrated into a Rails 7 application using importmaps. This isn't a theoretical exercise for me; I remember wrestling (oops, almost said it!) with this exact issue back when I was migrating a large legacy app over to the new asset pipeline in Rails 7. The shift from Webpacker was… well, let's just say it presented some interesting challenges. Importmaps, at first glance, seem straightforward, but the devil is often in the details, especially when you're dealing with third-party libraries. The main aim here isn't just to get toastr working, but also to understand the underlying mechanism of how importmaps operate.

The core idea behind importmaps is that we’re essentially declaring a mapping between module specifiers (what we use in our javascript import statements) and the actual locations of the javascript files. This eliminates the need for a traditional bundling process during development. Instead, the browser directly fetches the javascript files based on these mappings. This greatly simplifies the setup and reduces build times.

Now, before diving into code, it's crucial to understand that `importmap.rb` is the central configuration file. We’ll need to declare toastr and any of its dependencies (if there are any) here. Also, understand this is not necessarily limited to the CDN. A local copy of the library can be used, which I’ll show. It's also important to understand that toastr depends on jquery in older versions; therefore, we will use an implementation with a jquery dependency as an example. There are different implementations that do not, and these follow the same pattern, only without the need for the jquery mapping.

Here’s a typical `importmap.rb` snippet that incorporates toastr (assuming a CDN is being used).

```ruby
pin "application", preload: true
pin "@hotwired/turbo-rails", to: "turbo.min.js", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin "@hotwired/stimulus-loading", to: "stimulus-loading.js", preload: true
pin_all_from "app/javascript/controllers", under: "controllers"

pin "jquery", to: "https://code.jquery.com/jquery-3.7.1.min.js", preload: true
pin "toastr", to: "https://cdnjs.cloudflare.com/ajax/libs/toastr.js/latest/js/toastr.min.js", preload: true

```

Here's what this section is doing:

*   `pin "jquery" ...`: This line defines the mapping for jquery, pointing directly to a CDN hosted copy. The `preload: true` attribute hints that it's beneficial to load it early for performance.
*   `pin "toastr" ...`: This does the same for the toastr library, locating its minified javascript file on a CDN.

This configuration tells the browser, "whenever you see `import 'jquery'` or `import 'toastr'`, go to these specific URLs and fetch the JavaScript." Keep in mind that the browser relies on these exact module specifier names.

Now, in your javascript (likely in `app/javascript/application.js` or a related file):

```javascript
import 'jquery';
import 'toastr';

document.addEventListener('DOMContentLoaded', function() {
  toastr.success('Hello, toastr!');
});
```

This uses the aliases that we established. The import statements pull in the javascript and subsequently are made available via the global scope by `toastr.js`, allowing us to call `toastr.success()` once the document is ready.

While a CDN makes the setup incredibly straightforward, there may be situations where you need greater control and prefer a local copy of the library or are required to due to security protocols. You can download the toastr and jQuery javascript file, and then save this, for instance, in your `vendor` directory. Then, assuming that you create the directory structure `/vendor/javascript`, your `importmap.rb` would be changed to this:

```ruby
pin "application", preload: true
pin "@hotwired/turbo-rails", to: "turbo.min.js", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin "@hotwired/stimulus-loading", to: "stimulus-loading.js", preload: true
pin_all_from "app/javascript/controllers", under: "controllers"

pin "jquery", to: "vendor/javascript/jquery-3.7.1.min.js"
pin "toastr", to: "vendor/javascript/toastr.min.js"
```

And the javascript import statements in the javascript file (`app/javascript/application.js` or related file) remain the same:

```javascript
import 'jquery';
import 'toastr';

document.addEventListener('DOMContentLoaded', function() {
  toastr.success('Hello, toastr!');
});
```

The change is that we have pointed to a local file system rather than a CDN. This means that if the internet connection is interrupted, the javascript files will still be served.

Now, let’s think about the css. This is not handled by importmaps, as these focus on javascript files. The toastr css can be included via a stylesheet tag in the layout or via a css file added to the `app/assets/stylesheets` directory, and that is beyond the scope of importmaps. However, it is a key thing to make note of. If you use the local files, you would download the css and save it to the stylesheet directory. If you use the CDN, then you would include the stylesheet in the layout as below:

```html
<link href="https://cdnjs.cloudflare.com/ajax/libs/toastr.js/latest/css/toastr.min.css" rel="stylesheet">
```

In my experience, a common pitfall is overlooking these dependency declarations in `importmap.rb`. I’ve seen a lot of developers, myself included at the beginning, wonder why their imports aren’t working, only to find out that the modules weren't properly mapped in the importmap configuration. Another common problem is forgetting to include the css file, leading to functionality without styling.

The key takeaway here is that importmaps aren't meant to be a replacement for bundling, but an alternative during development, which can be a major benefit if you are in a large organization with a significant development and staging environment. If you are going to deploy a production version, it is highly recommended that you use a bundler such as webpack or esbuild to prepare a single package.

For further detailed learning, I would strongly recommend diving into these resources:

*   **The official Rails guides on importmaps**: These are updated regularly, are authoritative and provide excellent clarity on best practices.
*   **"JavaScript for impatient programmers" by Dr. Axel Rauschmayer**: This is an incredibly comprehensive text on all things JavaScript, and it gives a far deeper understanding of the module system than I could describe here.
*   **The HTML standard**: specifically section on importmaps. This can be useful if you want a full understanding of the inner workings of the feature.

The importmap approach can simplify your development setup significantly and, based on my past experiences, offers a more direct approach to dependency management than traditional bundlers, if your project is suitable. Just remember to keep your mappings accurate, your css included, and be aware of the need for bundling during production. It can be a game-changer for development velocity. And that, in a nutshell, is how you can use importmaps with toastr in Rails 7.
