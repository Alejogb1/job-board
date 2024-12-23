---
title: "How to use view_component inside a stimulus controller?"
date: "2024-12-23"
id: "how-to-use-viewcomponent-inside-a-stimulus-controller"
---

Alright,  It’s a question I've seen come up in various contexts, and I recall struggling with it myself years ago on a project involving a complex multi-panel interface. We were migrating from a monolithic server-rendered approach to a more componentized front-end, and the seamless integration between stimulus and view_component was… not immediately obvious. So, let’s break down how to effectively use view_component inside a stimulus controller.

The core issue centers around the inherent separation of concerns between these two powerful tools. `view_component` focuses on server-side rendering of reusable ui elements, while `stimulus` is all about adding dynamic behavior to the client-side. The challenge arises when you need to use stimulus’s event handling capabilities to interact with elements rendered by your view component. You can't just attach a controller to a component's root element and expect everything to function flawlessly, at least not without some intermediary steps.

The first point to understand is the lifecycle. A `view_component` renders HTML on the server. The browser then interprets this HTML and renders it to the DOM. Afterward, `stimulus` controllers are initialized, finding their targets based on specified `data-controller` attributes. The problem appears if your view component contains the `data-controller` attribute within the markup, because it might be rendering content that stimulus needs to react to dynamically. This means if the component is rerendered, stimulus might lose its connection to that DOM, which is why simply attaching a stimulus controller directly to a view component root element isn’t usually the most robust method. We require a slightly different pattern.

The key is leveraging a container element and `stimulus`’s ability to modify or re-render content. I’ve found that the most effective approach is to have the view component render the structural HTML (the wrapper and any static content) and then use stimulus to manage interactive parts within that structure. Here's an illustration of what I mean:

**Example 1: Simple Counter**

Imagine you have a view component rendering a counter. Instead of placing the `data-controller` on the root of that component, render a container and let the stimulus controller initialize the counter functionality within that wrapper.

```ruby
# app/components/counter_component.rb
class CounterComponent < ViewComponent::Base
  def initialize(initial_count: 0)
    @initial_count = initial_count
  end

  def call
    render(template)
  end

  def template
    <<~HTML
      <div class="counter-container">
        <span class="counter-value">#{@initial_count}</span>
        <button data-action="counter#increment">+</button>
      </div>
    HTML
  end
end
```

```javascript
// app/javascript/controllers/counter_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["value"];

  increment() {
    const currentValue = parseInt(this.element.querySelector('.counter-value').textContent, 10);
    this.element.querySelector('.counter-value').textContent = currentValue + 1;
  }
}
```

```erb
<!-- app/views/pages/some_page.html.erb -->
<div data-controller="counter">
  <%= render CounterComponent.new(initial_count: 5) %>
</div>
```

In this scenario, the root `div` in the ERB template holds the `data-controller="counter"`, while the `counter_component` only renders static HTML content. The controller then grabs the `span` element and attaches the action on the button.

**Example 2: Dynamically Updating a List**

Let’s consider a more complex example. Suppose you need to update a list of items within a view component based on an interaction. Here, we’ll use stimulus to request an updated list from the server, then re-render the list using a templating approach, making the update dynamic without refreshing the whole page:

```ruby
# app/components/item_list_component.rb
class ItemListComponent < ViewComponent::Base
  def initialize(items:)
    @items = items
  end

  def call
    render(template)
  end

  def template
    <<~HTML
    <div class="item-list-container" data-item-list-target="list">
      <ul data-item-list-target="list">
        #{ render_items_list }
      </ul>
      <button data-action="item-list#updateList">Update List</button>
    </div>
    HTML
  end

  def render_items_list
      @items.map { |item| "<li>#{item}</li>" }.join.html_safe
  end
end
```

```javascript
// app/javascript/controllers/item_list_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["list"];

  async updateList() {
      try {
        const response = await fetch('/items.json'); // Assuming you have an endpoint that returns items
        const data = await response.json();
        const newListItemsHTML = data.map(item => `<li>${item}</li>`).join('');
        this.listTarget.innerHTML = newListItemsHTML;
      } catch(error) {
         console.error('Error fetching or updating items', error)
      }
  }
}
```

```erb
<!-- app/views/pages/another_page.html.erb -->
<div data-controller="item-list">
  <%= render ItemListComponent.new(items: ["Item 1", "Item 2", "Item 3"]) %>
</div>
```

Here, `item_list_component` renders the initial list, and `item_list_controller` fetches new items and updates the DOM accordingly. Crucially, the view component only produces the markup, and stimulus handles the dynamic behavior.

**Example 3: Using Stimulus for Input Control in a Component Form**

Finally, let's consider a situation where you have form elements within a view component, and you need to validate input as the user types, or perform some type of transformation.

```ruby
# app/components/form_component.rb
class FormComponent < ViewComponent::Base
  def call
    render(template)
  end

  def template
    <<~HTML
    <div class="form-container">
      <label for="user-email">Email:</label>
      <input type="email" id="user-email" data-form-target="email" data-action="blur->form#validateEmail" >
      <span data-form-target="emailError" class="error-message" style="display: none;">Invalid Email</span>
      <button>Submit</button>
    </div>
    HTML
  end
end
```

```javascript
// app/javascript/controllers/form_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
    static targets = ["email", "emailError"];
    validateEmail() {
        const email = this.emailTarget.value;
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if(!emailRegex.test(email)){
          this.emailErrorTarget.style.display = 'inline';
          return;
        }
        this.emailErrorTarget.style.display = 'none';
    }
}
```

```erb
<!-- app/views/pages/yet_another_page.html.erb -->
<div data-controller="form">
  <%= render FormComponent.new %>
</div>
```

In this final example, the `form_component` provides the form structure. We can then control the email validation logic via the form controller, hiding or showing validation errors based on the user’s input. Notice the use of `data-action="blur->form#validateEmail"` – this lets us target the `validateEmail` action on a blur event of that input.

In essence, the approach revolves around not letting view components own the dynamic parts. Instead, treat the view component as your static markup renderer, and then leverage stimulus controllers to add interaction and behavior to your application elements inside the view component's generated html.

For deeper dives, I recommend exploring these resources:

*   **"Component-Based Rails" by Noel Rappin:** This book provides a deep understanding of how to structure large Rails applications with components, which directly relates to the thinking required to handle view components.

*   **The official Stimulus Handbook:** It offers detailed explanations of controller lifecycles, targets, and actions – fundamental concepts when integrating with other technologies such as view component.

*   **"Refactoring UI" by Adam Wathan and Steve Schoger:** While not directly code-focused, this book helps you think more about component design and structure, which will translate into cleaner integration with stimulus.

By embracing this architectural pattern, you can harness the power of both `view_component` and `stimulus` effectively, building scalable and maintainable web applications. This decoupling of static rendering from dynamic behavior is key to avoiding common pitfalls.
