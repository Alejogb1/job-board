---
title: "How do I set up Plausible Analytics with Rails 7 Importmaps?"
date: "2024-12-16"
id: "how-do-i-set-up-plausible-analytics-with-rails-7-importmaps"
---

Okay, let's tackle this. Integrating Plausible Analytics with a Rails 7 application using Importmaps, it's a setup I've navigated before, and frankly, it's not as straightforward as just dropping a script tag into your layout, due to how Importmaps manage dependencies. I remember struggling with this on a project a couple of years back when we were migrating from Webpacker, and the key is understanding how Importmaps handle external libraries and how to hook into that pipeline. Let's break it down.

The core challenge lies in the fact that Importmaps don't directly process JavaScript files; they simply map module specifiers to their corresponding locations. Unlike Webpacker, which bundles all your assets together, Importmaps rely on the browser's native module loading capabilities. This means Plausible's script, which you typically include via a `<script>` tag, needs to be handled differently.

First, you'll need to define Plausible's script location within your `config/importmap.rb` file. While Plausible doesn't offer a traditional npm package, we can still use their hosted script. Here's how I usually structure it in my import map:

```ruby
# config/importmap.rb
pin "application", preload: true
pin "@hotwired/turbo", to: "turbo.min.js", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin "@hotwired/stimulus-loading", to: "stimulus-loading.js", preload: true
pin_all_from "app/javascript/controllers", under: "controllers"
pin "plausible", to: "https://plausible.io/js/plausible.js", preload: true
```

The `pin "plausible", to: ...` line is the crucial part here. We're explicitly telling the import map to associate the "plausible" module specifier with the remote Plausible script. The `preload: true` flag is optional, but I generally prefer to include it for performance to enable browsers to preload this dependency early. Note that we're not copying this script into our project; instead, we’re pointing directly to its location on the Plausible CDN.

Now, with that defined, we can actually utilize this module in our `app/javascript/application.js` file, or within another relevant controller file that might be more suitable for your structure. Here's an example of how I normally initialise Plausible when the page loads:

```javascript
// app/javascript/application.js

import * as Turbo from "@hotwired/turbo"
import { Application } from "@hotwired/stimulus"
import { definitionsFromContext } from "@hotwired/stimulus-webpack-helpers"
import "plausible"

// Check if the Plausible script has loaded successfully before proceeding
document.addEventListener("DOMContentLoaded", () => {
    // check the global `plausible` var before attempting to access it
    if (typeof plausible !== "undefined") {
        // It's always good practice to check if an element exists on the page before trying to interact with it
        // You might not want to send analytics on every page in a typical application.
        if (document.querySelector("[data-plausible-domain]")) {
            // The domain will be passed by a view
            const domain = document.querySelector("[data-plausible-domain]").dataset.plausibleDomain;
            plausible('pageview', {
              props: {
                domain: domain
              }
            });
        }
    }
});

window.Stimulus = Application.start()
const context = require.context("./controllers", true, /\.js$/)
Stimulus.load(definitionsFromContext(context))

```

In this example, we import the 'plausible' module, which, due to our `importmap.rb` configuration, triggers the loading of the script from Plausible's CDN. The key improvement here is wrapping the plausible call in a document.addEventListener to ensure that the script, as well as the document are fully loaded. Additionally, the conditional on typeof `plausible` ensures we don’t get errors if the script fails to load for some reason. Finally, the check for an element with the `data-plausible-domain` attribute allows you to trigger analytics selectively depending on your application’s requirements.

Now, let's expand on a more advanced usage scenario, such as tracking a custom event. Imagine we have a button element in our view that we want to track clicks on. A typical Rails way to accomplish that would be using Stimulus. Here's how that might look:

```javascript
// app/javascript/controllers/click_tracker_controller.js

import { Controller } from "@hotwired/stimulus"
import "plausible"

export default class extends Controller {
    connect() {
      this.element.addEventListener('click', this.trackClick.bind(this))
    }

    trackClick() {
      if (typeof plausible !== "undefined") {
        // Data attribute `data-event-name` would be used in the view
          const eventName = this.element.dataset.eventName
          plausible(eventName);
      }
    }
}
```

Here, we have a Stimulus controller that attaches a click event listener to the tracked button. When the button is clicked, it calls the `plausible` function with the event name. Again, we’re verifying that `plausible` is defined before we call it to prevent errors. We'd then tie this controller to our button with a `data-controller` attribute and pass the event name through the `data-event-name` attribute in the view.

Finally, to make use of this, in your Rails view, you'd need to include a similar attribute, perhaps like so:

```erb
<div data-plausible-domain="<%= your_plausible_domain %>">
  <button data-controller="click-tracker" data-action="click->click-tracker#trackClick" data-event-name="submit_button_clicked">Click me</button>
</div>
```

Here, the `data-plausible-domain` is passed through to be used in the initial `application.js` plausible call. The button is targeted by the `click_tracker_controller`.

It’s crucial to understand that Importmaps are designed for straightforward dependency management. If you find yourself needing more complex bundling and transformation operations, you might want to explore other tooling options that complement Importmaps or move away from them altogether. However, this approach using remote sources with Importmaps is an excellent solution for many use-cases such as Plausible Analytics.

For further reading on this topic, I recommend exploring the official documentation for Rails Importmaps, as it covers the general principles quite thoroughly. Also, take a look at "JavaScript for Web Developers" by Nicholas C. Zakas; it provides excellent context on how JavaScript module loading works in the browser, which will deepen your understanding of how Importmaps function. Finally, "Eloquent JavaScript" by Marijn Haverbeke offers invaluable insights into the core concepts of the language that will prove helpful when dealing with any complex JavaScript interactions. These resources will provide you with a strong foundation for building well-structured JavaScript applications with Rails and Importmaps, and how they integrate with external libraries.
