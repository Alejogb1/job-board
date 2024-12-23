---
title: "How do I load page-specific custom JavaScript functions in Rails 7 with Turbo, ImportMaps and Stimulus?"
date: "2024-12-23"
id: "how-do-i-load-page-specific-custom-javascript-functions-in-rails-7-with-turbo-importmaps-and-stimulus"
---

Alright,  Having spent quite a bit of time migrating some legacy Rails applications to the modern stack with Turbo, ImportMaps, and Stimulus, I've definitely seen a few approaches for handling page-specific JavaScript. It’s a common challenge, ensuring that the right bits of client-side logic are present only when and where they’re needed. The key here is to avoid a monolithic JavaScript file, something I've learned the hard way.

The core of the issue revolves around the lifecycle changes introduced by Turbo. No longer do we have full page reloads on navigation. Instead, Turbo morphs the document, and this means any directly attached event listeners and initializations we’d normally perform with document.ready don’t get re-applied. This is precisely why frameworks like Stimulus become invaluable for managing such dynamics.

First and foremost, we need to understand that ImportMaps, unlike webpacker or other bundlers, doesn't automatically load everything. You're explicitly defining what is available in the browser’s global namespace. Stimulus, on the other hand, is designed to work declaratively within the HTML using data attributes, which aligns very well with Turbo's morphing behavior.

So, how do we achieve page-specific JavaScript? Let’s break it down. We'll avoid using global scope directly as much as we can, and lean on Stimulus controllers to encapsulate the logic.

**Approach 1: Data Attributes and Stimulus Controllers**

The most robust and recommended method is to attach a Stimulus controller only to specific HTML elements within the view of the page needing that functionality. This way, when Turbo morphs the content, the Stimulus controller will be activated or deactivated as necessary. I've found this to be the most reliable strategy across several projects.

Here’s how the code typically looks. Suppose you have a page that needs a custom datepicker component:

**1. HTML View (example: `app/views/my_resource/edit.html.erb`)**

```erb
<div data-controller="datepicker">
  <input type="text" data-datepicker-target="input">
</div>
```

**2. Stimulus Controller (example: `app/javascript/controllers/datepicker_controller.js`)**

```javascript
import { Controller } from "@hotwired/stimulus";
import Datepicker from "vanillajs-datepicker"; // assuming you have a datepicker library

export default class extends Controller {
    static targets = ["input"];

    connect() {
      this.datepicker = new Datepicker(this.inputTarget, {
      format: 'yyyy-mm-dd',
      autohide: true
      });
    }

    disconnect() {
      if (this.datepicker) {
        this.datepicker.destroy();
      }
    }
}
```

**3. ImportMap (example: `config/importmap.rb`)**

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin "vanillajs-datepicker", to: "vanillajs-datepicker.js", preload: true
pin_all_from "app/javascript/controllers", under: "controllers"
```

**Explanation:**

*   In the HTML view, we add the `data-controller="datepicker"` attribute to an element. Stimulus will automatically find all elements with this attribute on page load and any subsequent Turbo morph.
*   The Stimulus controller, `datepicker_controller.js`, initializes the datepicker when the controller connects and destroys it upon disconnect. This connection/disconnection process perfectly manages the lifecycle of the datepicker as Turbo navigates.
*   The `importmap.rb` file defines how the javascript modules are mapped for use within the browser’s global scope. You can pin the controllers directory and it will make all of the files within that folder available as controllers when they are referenced.

**Approach 2: Conditionals Within the Controller**

Sometimes, the logic might not be specific to a single element. In these scenarios, we can use conditionals within a Stimulus controller. This approach can be suitable for situations where you need to run a script on a page only if a specific element (that’s not necessarily used as the target of the controller) exists within the view.

**1. HTML View (example: `app/views/my_resource/index.html.erb`)**

```erb
<div data-controller="my-page-controller">
  <div id="special-container" style="display:none">
    <!-- content relevant to special page -->
  </div>
</div>
```

**2. Stimulus Controller (example: `app/javascript/controllers/my_page_controller.js`)**

```javascript
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  connect() {
    if(document.getElementById('special-container')){
        //page-specific logic here
        console.log("special container found!")
    }
  }
}
```

**3. ImportMap (same as the previous approach)**

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin_all_from "app/javascript/controllers", under: "controllers"
```

**Explanation:**

*   Here, we still have a Stimulus controller, but it’s applied to a more generic element.
*   The logic inside the controller checks for the presence of a specific element (`#special-container`). If that element exists, the script logic will be executed. This provides a page-specific condition without requiring a specific target within the controller’s scope.

**Approach 3: Using Data Attributes as Flags**

Lastly, you could leverage data attributes as flags directly on the body tag. This isn’t my typical go-to, as I generally prefer controllers on the elements using the javascript. Still, if you need to control things more directly at the page-level, it’s an option.

**1. Layout (example: `app/views/layouts/application.html.erb`)**

```erb
<body data-controller="global" data-page="<%= controller_name %>" data-action="<%= action_name %>" >
   <%= yield %>
</body>
```

**2. Global Stimulus Controller (example: `app/javascript/controllers/global_controller.js`)**

```javascript
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  connect() {
     const page = this.element.dataset.page;
     const action = this.element.dataset.action;
     // do something based on page or action
     if(page === "my_resource" && action === "edit"){
         console.log("This is the edit page!");
     }
  }
}
```

**3. ImportMap (same as the previous approach)**

```ruby
pin "application", preload: true
pin "@hotwired/stimulus", to: "stimulus.min.js", preload: true
pin_all_from "app/javascript/controllers", under: "controllers"
```

**Explanation:**

*   We attach the `global` controller to the `body` tag and include `controller_name` and `action_name` as data attributes.
*   The controller then retrieves these values and conditions upon them, making the javascript logic specific to the page and action that is currently being displayed.

**Important Considerations:**

*   **Avoiding Global Scope:**  As much as possible, keep code out of the global scope and use Stimulus controllers to encapsulate the logic.
*   **Lazy Loading:** Importmaps automatically handle a form of code splitting, ensuring only the code actually used on the page is loaded. You can also optimize the order of javascript imports within the importmap to ensure files are loaded appropriately.
*   **Resource Management:** Stimulus controllers handle connection and disconnection of elements. This is critical to prevent memory leaks and other issues that can occur from incorrectly attached event listeners.
*   **Modularization:** Try to break out code into smaller, more reusable Stimulus controllers. This makes code easier to understand and maintain.

**Recommended Resources:**

*   **"Programming Phoenix LiveView"** by Bruce A. Tate and Sophie DeBenedetto. While focusing on LiveView, this book provides fantastic insights into handling interactive client-side logic within modern web applications. The core principles apply to Stimulus and Turbo.
*   **The official Stimulus Documentation:** It provides very helpful resources and is well organized.
*   **The Hotwire Handbook:** A guide to using Turbo and Stimulus.
*   **The Rails Documentation:** Specifically the sections on ActionView, Assets, and Hotwire.
*   **“Eloquent JavaScript”** by Marijn Haverbeke: A fundamental book on core javascript principles that helps deepen understanding of the javascript language and its patterns.

In summary, moving to Turbo, ImportMaps, and Stimulus means moving to a component-based model with clear lifecycle management. You’ll find that leveraging Stimulus controllers, along with clear patterns for attaching them to page-specific elements (or conditioning upon the element’s presence within the controller’s logic), will be the most reliable method for loading page-specific JavaScript in your Rails 7 application. It took me a couple of iterations to get comfortable with these approaches, but it's definitely worth the effort for maintainability and overall robustness.
