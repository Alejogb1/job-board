---
title: "How to load page-specific custom JS in Rails 7 with Turbo/ImportMaps/Stimulus?"
date: "2024-12-16"
id: "how-to-load-page-specific-custom-js-in-rails-7-with-turboimportmapsstimulus"
---

Alright, let's tackle this. It’s a question I've spent more than a few late nights wrestling with, especially in the early days of transitioning legacy Rails apps to Turbo. The challenge of loading page-specific javascript in a modern rails 7 application using turbo, importmaps, and stimulus is definitely non-trivial, but it’s far from insurmountable. Instead of the old way with rails asset pipeline, we've now got importmaps for module resolution and stimulus for controller behavior, so we need to think differently about how to structure our javascript code.

The traditional asset pipeline often relied on server-side logic to generate page-specific javascript includes. With Turbo and importmaps, we lean more towards a client-side approach for handling such scenarios. Importmaps centralize javascript module definitions, and Turbo enhances the application experience by delivering partial page updates, so our scripts now need to load dynamically. This means we have to be selective about what scripts we load and when they're executed.

The core idea here is to use stimulus controllers, coupled with some clever use of data attributes on our HTML, to figure out when and what modules to import on a per-page basis. Let’s think through this with some practical examples.

**The Core Strategy: Data Attributes and Stimulus Controllers**

The general approach I’ve found consistently effective involves a combination of data attributes in your views and a dispatcher stimulus controller. The dispatcher listens for turbo events and then uses the attributes to identify what javascript to load and which stimulus controllers to attach.

Let's imagine that we've got a blog application. We might have a different set of javascript behaviors on the `posts#show` view versus the `posts#index` view. In the past, we might have used rails asset pipeline and some `if` logic on server side. Now, we’ll use stimulus and data attributes.

Here's how it might work with some sample code.

**Example 1: Basic Page-Specific Script Loading**

First, let's define our `dispatcher_controller.js`:

```javascript
// app/javascript/controllers/dispatcher_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {
  connect() {
    document.addEventListener("turbo:load", this.loadModules.bind(this));
    this.loadModules(); //initial load
  }

  disconnect() {
    document.removeEventListener("turbo:load", this.loadModules.bind(this));
  }

  loadModules() {
    const pageScripts = this.element.dataset.pageScripts;
    if (pageScripts) {
      pageScripts.split(' ').forEach(async (script) => {
        try {
          await import(`../controllers/${script}_controller`);
          console.log(`Loaded controller: ${script}`);
        } catch (error) {
          console.error(`Error loading controller ${script}:`, error);
        }
      });
    }

    const pageStimulus = this.element.dataset.pageStimulus;
      if(pageStimulus) {
        pageStimulus.split(' ').forEach(controllerName => {
            const controllerElement = document.querySelector(`[data-controller~="${controllerName}"]`);

             if (controllerElement) {
                if(this.application.getControllerForElementAndIdentifier(controllerElement, controllerName)) return;

                this.application.register(controllerName, require(`../controllers/${controllerName}_controller`).default);
                console.log(`Registered Controller: ${controllerName}`)
            }

        });
    }
  }
}
```

Here's how our `posts/show.html.erb` might look:

```erb
<div data-controller="dispatcher" data-page-scripts="post_show" data-page-stimulus="post_carousel">
    <h1><%= @post.title %></h1>
    <div data-controller="post_carousel">
      <%# Carousel code here %>
    </div>
</div>
```

And then create the javascript files:

```javascript
// app/javascript/controllers/post_show_controller.js
export default class extends Controller {
    connect() {
        console.log("Post Show controller connected");
        // Add logic here for post specific behaviors
    }
}
```
```javascript
// app/javascript/controllers/post_carousel_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
    connect() {
        console.log("Post Carousel controller connected")
    }
}
```

In this setup, the `dispatcher_controller` loads the `post_show_controller` when the `posts#show` view is rendered (either via initial visit or a turbo navigation). It also looks for stimulus controllers defined in the `data-page-stimulus` attribute and then loads them on page load.

**Example 2: Handling Dynamic Content**

Let's extend this with a scenario where content is dynamically loaded via a turbo stream. For this, we need to account for newly added DOM elements. Let's assume that when a comment is added to a post we get a turbo stream replacing a content placeholder. We then would need the relevant javascript to attach to the newly rendered HTML. Here is our updated dispatcher controller:

```javascript
// app/javascript/controllers/dispatcher_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {
  connect() {
    document.addEventListener("turbo:load", this.loadModules.bind(this));
    document.addEventListener("turbo:frame-load", this.handleFrameLoad.bind(this))
      this.loadModules(); //initial load
  }

  disconnect() {
    document.removeEventListener("turbo:load", this.loadModules.bind(this));
    document.removeEventListener("turbo:frame-load", this.handleFrameLoad.bind(this));
  }

  handleFrameLoad(event){
    const element = event.detail.element;
    this.loadModules(element);
  }

  loadModules(element = document) {
    const pageScripts = element.dataset?.pageScripts;
    if (pageScripts) {
      pageScripts.split(' ').forEach(async (script) => {
        try {
          await import(`../controllers/${script}_controller`);
          console.log(`Loaded controller: ${script}`);
        } catch (error) {
          console.error(`Error loading controller ${script}:`, error);
        }
      });
    }

    const pageStimulus = element.dataset?.pageStimulus;
      if(pageStimulus) {
        pageStimulus.split(' ').forEach(controllerName => {
          const controllerElement = element.querySelector(`[data-controller~="${controllerName}"]`);

             if (controllerElement) {
              if(this.application.getControllerForElementAndIdentifier(controllerElement, controllerName)) return;

                this.application.register(controllerName, require(`../controllers/${controllerName}_controller`).default);
                console.log(`Registered Controller: ${controllerName}`)
            }
        });
    }
  }
}
```

And now, `posts/show.html.erb`:

```erb
<div data-controller="dispatcher" data-page-scripts="post_show" data-page-stimulus="post_carousel comment_form">
    <h1><%= @post.title %></h1>
    <div data-controller="post_carousel">
      <%# Carousel code here %>
    </div>
     <div id="comment_area">
          <%= render "comments/form", post: @post %>
      </div>
</div>

```

And here is the `_comment_form.html.erb` partial

```erb
<div id="<%= dom_id(Comment.new)%>" data-controller="comment_form" data-page-stimulus="comment_form">
      <%= form_with model: Comment.new, url: post_comments_path(@post), data: { turbo_frame: "comment_area"} do |f|%>
         <%= f.text_area :body %>
         <%= f.submit "Add Comment"%>
      <% end %>
</div>
```

The comment form is now contained in a `turbo-frame` so after the form submission, only the comment area will be updated. The dispatcher controller registers itself and now can handle both normal turbo-loads and turbo-frame-loads. When a new form is loaded the dispatcher loads the `comment_form_controller` and ensures it is registered for the form.

**Example 3: Complex Conditional Logic**

Sometimes we need more complex rules. Let's say we have an admin area where we want to load different scripts based on the current user role. We’ll keep the data attributes but define our dynamic script loading inside the `loadModules` method.

```javascript
// app/javascript/controllers/dispatcher_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {
  connect() {
    document.addEventListener("turbo:load", this.loadModules.bind(this));
    document.addEventListener("turbo:frame-load", this.handleFrameLoad.bind(this));
    this.loadModules(); // initial load
  }

  disconnect() {
    document.removeEventListener("turbo:load", this.loadModules.bind(this));
    document.removeEventListener("turbo:frame-load", this.handleFrameLoad.bind(this));
  }


  handleFrameLoad(event){
      const element = event.detail.element;
      this.loadModules(element);
    }

  async loadModules(element = document) {
    const pageScripts = element.dataset?.pageScripts;

    if (pageScripts) {
        const scripts = pageScripts.split(' ');
      for (const script of scripts) {
        try {
           await import(`../controllers/${script}_controller`);
           console.log(`Loaded controller: ${script}`);
        }
        catch(e) {
          console.error(`Error loading controller ${script}:`, e);
        }
      }
    }

    const pageStimulus = element.dataset?.pageStimulus;
      if(pageStimulus) {
        pageStimulus.split(' ').forEach(controllerName => {
          const controllerElement = element.querySelector(`[data-controller~="${controllerName}"]`);

             if (controllerElement) {
              if(this.application.getControllerForElementAndIdentifier(controllerElement, controllerName)) return;

                this.application.register(controllerName, require(`../controllers/${controllerName}_controller`).default);
                console.log(`Registered Controller: ${controllerName}`)
            }
        });
    }
  }
}
```

And now in our layout or view:
```erb
<div data-controller="dispatcher" data-page-scripts="<%= current_user.admin? ? 'admin_scripts' : 'user_scripts'%>" data-page-stimulus="default_stimulus">
  <%= yield %>
</div>
```

In this version, the `page-scripts` attribute is dynamically generated based on the user’s role (you might do this via a helper method). This allows us to dynamically import scripts like `admin_scripts_controller` or `user_scripts_controller` as needed.

**Key Takeaways and Further Reading**

*   **Turbo Events:** Understanding `turbo:load` and `turbo:frame-load` is fundamental for this approach. The official Turbo documentation, especially the section on event handling, is a must-read.
*   **Importmaps:** Familiarize yourself with how importmaps work for module resolution. Refer to the Rails Guides on asset pipelines and JavaScript bundlers for in-depth information about how this integrates with Rails 7.
*   **Stimulus Fundamentals:** A strong grasp of how Stimulus controllers work is necessary. The official Stimulus documentation is indispensable for learning more about controllers, their lifecycle, and how to effectively use them.

The examples provided should provide a clear path forward. It involves understanding the interplay between Turbo, importmaps, and stimulus, and how these components can help load the proper javascript in a clean and maintainable way. Remember that every application's needs are unique, so you’ll need to adapt this approach to fit your specific requirements. By embracing this more dynamic client side loading approach, you'll find that modern javascript in a rails application is powerful and allows a significant gain in flexibility.
