---
title: "How do I setup Plausible Analytics with Rails 7 Importmaps?"
date: "2024-12-23"
id: "how-do-i-setup-plausible-analytics-with-rails-7-importmaps"
---

Okay, so let’s tackle integrating Plausible Analytics with a Rails 7 application using Importmaps. This isn’t exactly uncharted territory; I’ve navigated this type of integration several times over the years, and there are definitely some nuances to consider. My experiences, particularly with a large-scale e-commerce platform I worked on a few years back, revealed that getting this just so requires a deep understanding of how Importmaps function within the Rails asset pipeline. So, let's get to it.

The core challenge lies in the way Importmaps manage JavaScript dependencies. Instead of relying on Node.js and `npm` or `yarn`, Importmaps allow you to specify directly where JavaScript modules should be fetched from, typically a CDN, and how to import them into your project. This is more lightweight but requires careful configuration to avoid conflicts and ensure proper loading of Plausible's tracking script.

First, let’s examine the correct method to include the Plausible script. You wouldn't just dump it in a raw html tag. That’s a recipe for potential timing issues and conflicts. You want it to load cleanly, and through the asset pipeline, so that rails can take care of potential issues. The correct way is to import it, utilizing Importmaps.

Here's the step-by-step procedure I usually follow:

**1. Configure `importmap.rb`:**

This is the foundation. We need to inform rails that we wish to use plausible via it's script hosted on their cdn. Open your `config/importmap.rb` file, and add the following:

```ruby
pin "plausible", to: "https://plausible.io/js/script.js", preload: true
```

The key here is `pin "plausible", to: ...` . This tells Importmaps that when we import something as 'plausible' in our javascript, we are referencing the provided URL which contains the required Plausible JavaScript tracking script. The `preload: true` option is important here too. It instructs the browser to start downloading the script right away which improves user experience, since the script will be available almost instantly.

**2. Setting up the Application Javascript:**

Next, you need to include the library in your application-wide js file, commonly stored in `app/javascript/application.js`. It's important to include this in this manner, as importmaps requires usage of the module import syntax. Add the following to the file:

```javascript
import "./controllers" // rails default for controllers
import "plausible"
// any other required imports below this line.
```

By importing `plausible` , Importmaps now ensures that the Plausible script from the CDN is loaded and made available within our Rails application.

**3. Ensure Plausible Configuration is Correct:**

With this done, the script is technically running, but it may not be doing what you want it to do. It must be able to communicate with your Plausible installation, so we'll need to add the `data-domain` attribute to the `body` tag of the `application.html.erb` file of your app (or your specific view if you are just testing). The attribute must match the domain configured in your Plausible setup. This is a critical step. If this is not setup, your data will not be sent to your account on the Plausible servers.

Here's what that might look like:

```html+erb
<!DOCTYPE html>
<html>
  <head>
    <title>Your App Title</title>
    <%= csrf_meta_tags %>
    <%= csp_meta_tag %>

    <%= stylesheet_link_tag "application", "data-turbo-track": "reload" %>
    <%= javascript_importmap_tags %>
  </head>

  <body data-domain="yourdomain.com">
    <%= yield %>
  </body>
</html>
```

Make sure to replace `yourdomain.com` with the exact domain you have configured within your Plausible account.

**4. Handling Single Page Applications (SPAs) Considerations:**

If your Rails application leverages Hotwire or something similar that alters the page structure without full reloads, or utilizes a JS framework like React, Vue or Angular, you’ll need to manually trigger Plausible’s page view events upon navigation. Plausible automatically tracks pageviews on full page reloads, but not on the types of transitions that SPAs facilitate.

To do this, you can use Turbo’s events. Here's an example using Turbo's `turbo:load` event listener.

```javascript
// app/javascript/application.js

import "./controllers"
import "plausible"

document.addEventListener("turbo:load", () => {
    if (window.plausible) {
        window.plausible('pageview')
    }
});

```

This code snippet ensures that a pageview is tracked every time the page updates via Turbo, thereby preserving accurate analytics in dynamic interfaces. If you utilize a different system to manage your routes, or a framework, it's important that you setup the equivalent functionality within the event system of said framework.

**Working code examples:**

Here are some snippets that demonstrate the concepts above in context:

**Example 1: `importmap.rb`**

```ruby
# config/importmap.rb
pin "application", preload: true
pin "@hotwired/turbo-rails", to: "turbo.min.js", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin "@hotwired/stimulus-loading", to: "stimulus-loading.js", preload: true
pin_all from: "app/javascript/controllers", under: "controllers"

pin "plausible", to: "https://plausible.io/js/script.js", preload: true
```

**Example 2: `application.js`**

```javascript
// app/javascript/application.js
import "./controllers"
import "plausible"

document.addEventListener("turbo:load", () => {
    if (window.plausible) {
        window.plausible('pageview')
    }
});
```

**Example 3: `application.html.erb`**

```html+erb
<!DOCTYPE html>
<html>
  <head>
    <title>Your Application Title</title>
    <%= csrf_meta_tags %>
    <%= csp_meta_tag %>

    <%= stylesheet_link_tag "application", "data-turbo-track": "reload" %>
    <%= javascript_importmap_tags %>
  </head>

  <body data-domain="mywebsite.test">
    <%= yield %>
  </body>
</html>
```

**Resource Recommendations:**

*   **"The Rails 7 Way" by David Heinemeier Hansson**: For an exhaustive understanding of how Rails, in its most modern form, operates.
*   **"High Performance Browser Networking" by Ilya Grigorik**: A deep dive into how browsers fetch and load resources, critical for optimizing preload and general performance with external scripts like the one provided by Plausible.
*  **The official Turbo Documentation**: To understand how the turbo system in rails functions, and what events are available.

**Final Thoughts**

Setting up Plausible with Rails 7 using Importmaps is straightforward, but it requires a careful understanding of how both the rails asset pipeline and importmaps work. Avoid using direct script tags in your templates, since importmaps is a better way to manage dependencies. Ensure the `data-domain` attribute is correct, and implement manual page view tracking for SPAs as needed. By approaching it methodically, you can seamlessly integrate Plausible for robust analytics without encountering the common pitfalls. This should give you a very solid foundation.
