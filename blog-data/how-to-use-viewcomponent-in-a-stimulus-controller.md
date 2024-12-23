---
title: "How to use view_component in a stimulus controller?"
date: "2024-12-16"
id: "how-to-use-viewcomponent-in-a-stimulus-controller"
---

, let's delve into that. The intersection of `view_component` and stimulus controllers can sometimes feel a bit like navigating a new, albeit familiar, terrain. I recall a project a few years back where we needed highly interactive components within our Rails application. We initially went down the path of traditional partials, but the logic became unwieldy quickly. That’s where `view_component` paired with stimulus really began to shine, offering a much more maintainable architecture.

The core idea is this: `view_component` handles the presentation logic and provides a nice encapsulation of component rendering, while stimulus handles the user interaction and associated client-side behaviors. When these two are properly integrated, you get components that are not only well-structured but also highly dynamic.

The primary challenge isn't generally about technical feasibility, but rather about ensuring proper communication between these two different worlds. We need a way to leverage the rendering capabilities of `view_component` within the dynamic context of a stimulus controller without creating an entangled mess. The key is to minimize the direct interaction between stimulus and the rendering logic, favoring well-defined interfaces.

Here's the approach I typically take, broken down into steps:

First, your `view_component` should focus solely on rendering its output based on the data it receives, with no client-side interaction embedded within. We pass data into the `view_component` instance, and it renders the HTML. Consider this simple `CardComponent` as an example:

```ruby
# app/components/card_component.rb
class CardComponent < ViewComponent::Base
  def initialize(title:, content:, initial_expanded: false)
    @title = title
    @content = content
    @expanded = initial_expanded
  end
end
```

And the corresponding template:

```erb
# app/components/card_component.html.erb
<div class="card" data-controller="card">
  <h3 class="card-title" data-action="click->card#toggle"><%= @title %></h3>
  <div class="card-content <%= 'expanded' if @expanded %>" data-card-target="content">
    <%= @content %>
  </div>
</div>
```

Here, the view component itself is essentially 'dumb' in terms of interactivity. The `data-controller="card"` and `data-action` attribute hints at the stimulus controller that will handle the dynamic aspects, which we will detail shortly.

Second, we define the stimulus controller. Its primary responsibility is to manage the dynamic state of the component: toggling classes, managing animations, or any other client-side behavior needed. The controller is kept as agnostic as possible about the specific content being rendered.

Here's the associated stimulus controller:

```javascript
// app/javascript/controllers/card_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = [ "content" ]

  connect() {
    if(this.contentTarget.classList.contains('expanded')){
      this.expanded = true;
    } else {
      this.expanded = false;
    }
  }
  toggle() {
      this.expanded = !this.expanded;
      this.contentTarget.classList.toggle('expanded');
  }
}
```

The stimulus controller here isn't tied to any specific data passed into the `CardComponent`. It manages the state purely by interacting with the DOM.

Third, and this is crucial: how do you actually integrate this into your view? You can't just call `render CardComponent.new(...)` directly within your layout or partial if you want to use its data attributes that stimulus is expecting. Instead, render the component directly into the view you want. The data attributes defined within the `.erb` file will be automatically picked up by the controller, without having to specify extra stimulus options at rendering.

For instance, within a view file:

```erb
<%# app/views/pages/show.html.erb %>
<div>
  <%= render CardComponent.new(title: "First Card", content: "This is the content of the first card.", initial_expanded: true) %>
  <%= render CardComponent.new(title: "Second Card", content: "Content for the second card.") %>
</div>
```

The important detail here is that when the page is loaded, the stimulus controller will auto-connect to the `card` element it finds on the page due to the data-controller value. Thus, on page load, stimulus handles the logic defined in the `connect` method, and any event triggers such as `click` will handle the `toggle` method within the stimulus controller.

The key takeaway is separation of concerns. The `view_component` handles rendering, and the stimulus controller handles interactions. By keeping the responsibilities distinct, the overall system becomes more maintainable.

Let's consider a slightly more complex scenario. Suppose you need to update the content of a component via an AJAX request. The view component wouldn't directly handle the AJAX; that's the domain of the stimulus controller.

Here's an updated version of the `CardComponent`, modified to handle the loading scenario within the content:

```ruby
# app/components/card_component.rb
class CardComponent < ViewComponent::Base
  def initialize(title:, content:, initial_expanded: false, loading: false)
    @title = title
    @content = content
    @expanded = initial_expanded
    @loading = loading
  end
end
```

And the template, modified to incorporate loading state handling:

```erb
# app/components/card_component.html.erb
<div class="card" data-controller="card">
  <h3 class="card-title" data-action="click->card#toggle"><%= @title %></h3>
  <div class="card-content <%= 'expanded' if @expanded %>" data-card-target="content">
    <% if @loading %>
      <span data-card-target="loader">Loading...</span>
    <% else %>
      <%= @content %>
    <% end %>
  </div>
</div>
```

Then, let's create a new stimulus controller to handle loading external content, and modifying the `toggle` method:

```javascript
// app/javascript/controllers/card_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = [ "content", "loader" ]
  static values = { loadingUrl: String }

  connect() {
    if(this.contentTarget.classList.contains('expanded')){
      this.expanded = true;
    } else {
      this.expanded = false;
    }
  }

  toggle() {
      if (this.expanded) {
          this.contentTarget.classList.remove('expanded');
      } else {
        if(this.hasLoadingUrlValue){
          this.loadContent()
        } else {
          this.contentTarget.classList.add('expanded')
        }
      }
      this.expanded = !this.expanded;
  }

  async loadContent() {
    this.loaderTarget.textContent = "Loading...";
    try {
      const response = await fetch(this.loadingUrlValue)
      if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.text();
      this.contentTarget.innerHTML = data
    } catch (error) {
        console.error("Failed to load content", error)
        this.contentTarget.textContent = "Error loading content."
    } finally {
      this.loaderTarget.textContent = "";
      this.contentTarget.classList.add('expanded')
    }
  }
}
```

In this example, we are also using a `loadingUrl` attribute to define a resource for the stimulus controller to request when we toggle open the card. Here's how you might render a component with this feature:

```erb
<%# app/views/pages/show.html.erb %>
<div>
  <%= render CardComponent.new(title: "Async Card", content: "Placeholder content", loading: true, html_attributes: {'data-card-loading-url': '/data/async-content'}) %>
</div>
```

Notice the use of the `html_attributes` to pass the `data-card-loading-url` to the component. This is key; data attributes are passed from the view into the rendered html that will be picked up by the stimulus controller.

In summary, combining `view_component` and stimulus controllers effectively involves a careful separation of responsibilities. `view_component` handles the rendering logic, while stimulus manages dynamic behaviors. By focusing on clear interfaces, data attributes, and minimizing tight coupling, you create a maintainable and robust architecture. For more in-depth exploration, I recommend reviewing the official documentation for both libraries: the `view_component` documentation on the official gem repository and the official stimulus documentation, as well as consider research into design patterns for client-side architectures within the Rails environment. These resources provide a solid foundation for further learning and real-world application.
