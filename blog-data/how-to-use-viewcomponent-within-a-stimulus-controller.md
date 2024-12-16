---
title: "How to use view_component within a stimulus controller?"
date: "2024-12-16"
id: "how-to-use-viewcomponent-within-a-stimulus-controller"
---

Alright, let's tackle this. I've spent a fair bit of time integrating `view_component` and Stimulus in some quite intricate applications, and it's a combination that, while initially feeling a bit like fitting square pegs in round holes, can become very powerful with the correct approach. It's not entirely straightforward, because the core philosophies of the two libraries are quite different. `view_component` focuses on encapsulated rendering logic and component composition, while Stimulus is primarily about DOM manipulation and behavior. Getting them to play nicely means understanding their distinct roles and where they overlap.

The fundamental challenge arises from the fact that `view_component` renders HTML markup on the server-side (or potentially at build time, depending on your configuration), while Stimulus operates exclusively in the browser. Therefore, we can't directly instantiate a `view_component` *inside* a Stimulus controller. What we *can* do, and what is usually the most effective strategy, is to utilize Stimulus to control the behavior of a rendered `view_component` or update it through server responses. I’ll explain three common scenarios I’ve found myself in, with illustrative code examples.

**Scenario 1: Initializing a Component with Stimulus**

In this situation, imagine a `TabsComponent` that renders a set of tab headers and corresponding content panes. The visual aspect is handled by the component, but we need Stimulus to toggle the active tab and manage the display of its associated content.

```ruby
# app/components/tabs_component.rb
class TabsComponent < ViewComponent::Base
  def initialize(tabs:)
    @tabs = tabs
  end
end
```

```erb
<%# app/components/tabs_component.html.erb %>
<div data-controller="tabs">
  <ul class="tabs-nav">
    <% @tabs.each_with_index do |tab, index| %>
      <li data-action="click->tabs#selectTab" data-tabs-target="tab" data-tab-index="<%= index %>"
          class="<%= 'active' if index == 0 %>">
        <%= tab[:title] %>
      </li>
    <% end %>
  </ul>

  <div class="tabs-content">
    <% @tabs.each_with_index do |tab, index| %>
      <div data-tabs-target="content" class="<%= 'active' if index == 0 %>">
        <%= tab[:content] %>
      </div>
    <% end %>
  </div>
</div>
```

And the corresponding Stimulus controller:

```javascript
// app/javascript/controllers/tabs_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {
  static targets = ["tab", "content"];

  connect() {
      this.showTab(0); // Show first tab on initial load
  }

  selectTab(event) {
    const tab = event.currentTarget;
    const tabIndex = parseInt(tab.dataset.tabIndex);
    this.showTab(tabIndex);
  }

    showTab(index) {
        this.tabTargets.forEach((tab, i) => tab.classList.toggle('active', i === index));
        this.contentTargets.forEach((content, i) => content.classList.toggle('active', i === index));
    }
}
```
Here, the `TabsComponent` is responsible for generating the structure and initial markup. The Stimulus `TabsController` then adds dynamic behavior, managing CSS classes to toggle active states. This demonstrates a clear separation of concerns: templating in `view_component`, and interactive behavior through Stimulus.

**Scenario 2: Dynamically Updating Component Content**

In many scenarios, you'll want to dynamically update the content of a component based on user interactions or server-side events. This often involves making a fetch request, and then replacing the inner HTML of part of the component. Suppose we have a `UserProfileComponent` which displays user details, and we want to reload it when the user updates their profile.

```ruby
# app/components/user_profile_component.rb
class UserProfileComponent < ViewComponent::Base
  def initialize(user:)
    @user = user
  end
end
```

```erb
<%# app/components/user_profile_component.html.erb %>
<div data-controller="user-profile">
  <div data-user-profile-target="profileContent">
    <h1><%= @user.name %></h1>
    <p><%= @user.email %></p>
  </div>
  <button data-action="click->user-profile#refreshProfile">Refresh Profile</button>
</div>
```

The Stimulus controller for this would be:
```javascript
// app/javascript/controllers/user_profile_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {
  static targets = ["profileContent"];

    refreshProfile() {
      fetch('/profile') // Endpoint that returns the updated profile partial HTML.
        .then(response => response.text())
        .then(html => {
          this.profileContentTarget.innerHTML = html;
      })
      .catch(error => console.error("Error fetching updated profile", error));
    }
}
```
In this example, the `UserProfileComponent` renders the initial user data. The `user-profile` stimulus controller uses a simple fetch request to grab the new content from the '/profile' path, and then inserts it into the profileContent target element, effectively refreshing the component on demand. The key here is that the server endpoint should return the complete updated content for the partial.

**Scenario 3: Using Stimulus for Component-Specific Interactions**

Let's say you have a `ModalComponent` which displays modal dialogs, and you want to use stimulus to control the opening and closing. This may not require you to explicitly render content each time.

```ruby
# app/components/modal_component.rb
class ModalComponent < ViewComponent::Base
  def initialize(title:, content:, id:)
      @title = title
      @content = content
      @id = id
  end
end
```

```erb
<%# app/components/modal_component.html.erb %>
<div data-controller="modal" data-modal-id="<%= @id %>"  class="modal"  style="display:none">
  <div class="modal-content">
      <span class="close" data-action="click->modal#close">&times;</span>
      <h2><%= @title %></h2>
      <p><%= @content %></p>
  </div>
</div>
```

And the Stimulus Controller:

```javascript
// app/javascript/controllers/modal_controller.js
import { Controller } from "@hotwired/stimulus";

export default class extends Controller {

    static values = { id : String }
    connect() {
       this.element.id = this.idValue;
    }

    open(){
        this.element.style.display = "block";
    }

    close(){
        this.element.style.display = "none";
    }
}
```
In this scenario, a modal `view_component` renders the modal with a unique ID and initial content, initially hidden. A separate button will be needed to trigger the opening action, referencing that modal by its ID. The Stimulus `ModalController` only handles the visual logic.

These three scenarios cover the vast majority of practical situations you'll encounter when using `view_component` and Stimulus together. The common thread is treating `view_component` as the rendering engine, providing static structure and content, and Stimulus as the behavioral layer, adding dynamism and interactivity.

For further reading, I recommend exploring "Agile Web Development with Rails 7" by David Heinemeier Hansson, et al. for a comprehensive overview of the Rails framework, including a deep dive into the `view_component` library, and the official documentation of Stimulus. It is critical to thoroughly grasp how each piece operates before attempting such integrations. Also, consider looking into the "Component Driven UI" pattern, a concept which, although not a specific text, is fundamental to these practices. Understanding this concept deeply improves how you can separate concerns in a more scalable way.

Integrating `view_component` with Stimulus isn't about forcing them into a single paradigm, but about appreciating their strengths and leveraging them together. It requires careful planning and understanding each component's responsibility, but the result is much more maintainable and scalable.
