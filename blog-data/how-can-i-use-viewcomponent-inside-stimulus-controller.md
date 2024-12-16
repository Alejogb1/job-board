---
title: "How can I use view_component inside stimulus controller?"
date: "2024-12-16"
id: "how-can-i-use-viewcomponent-inside-stimulus-controller"
---

Okay, let's tackle this. It's a situation I've encountered more than a few times, and it’s a surprisingly nuanced problem when you dive into the specifics. Integrating `view_component` directly within a Stimulus controller isn't the typical use case, but it absolutely has its merits when you need dynamic and reusable UI elements. You’ll often find yourself in scenarios where static html templates controlled by Stimulus become too verbose, or when logic separation warrants moving rendering responsibilities to the server-side.

The crux of the issue is that `view_component` generates server-rendered html, while Stimulus operates entirely in the client-side. Therefore, we can’t just instantiate a `view_component` directly from within a stimulus controller. Instead, we need to bridge the gap, and the usual approach involves fetching the rendered component from the server and then manipulating the dom using the stimulus controller. I’ve seen this become complicated in projects where data fetching isn't handled cleanly and rendering logic gets mixed into the controllers. It's a maintenance nightmare waiting to happen.

My preferred method—and I've refined this over a few large-scale applications—is to rely on server-side endpoints that provide the html for specific component renderings based on data passed to them. This approach keeps the rendering logic centralized and allows the stimulus controller to act as an orchestrator, focusing on DOM updates and user interactions, as it was intended.

Let me illustrate this with a few snippets to make things more concrete.

**Example 1: Simple Component Replacement**

Let’s consider a scenario where you want to update a ‘user card’ component dynamically after an event, such as clicking a refresh button.

First, a simplified version of a `view_component`:

```ruby
# app/components/user_card_component.rb
class UserCardComponent < ViewComponent::Base
  def initialize(user:)
    @user = user
  end

  def template
    <<~HTML
      <div class="user-card">
        <h3>User: #{@user.name}</h3>
        <p>Email: #{@user.email}</p>
      </div>
    HTML
  end
end
```

Now, a route and controller action to render it based on user data:

```ruby
# config/routes.rb
  get 'users/:id/card', to: 'users#card', as: 'user_card'

# app/controllers/users_controller.rb
  def card
    @user = User.find(params[:id])
    render partial: "users/card", locals: { user: @user}
  end

# app/views/users/_card.html.erb
<%= render(UserCardComponent.new(user: user)) %>
```

Finally, the Stimulus controller:

```javascript
// app/javascript/controllers/user_card_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = [ "cardContainer" ]

  refreshCard() {
    const userId = this.element.dataset.userId; // Assuming the user id is on the element
    fetch(`/users/${userId}/card`)
      .then(response => response.text())
      .then(html => {
        this.cardContainerTarget.innerHTML = html;
      });
  }
}
```

and the HTML with Stimulus bindings:

```html
<div data-controller="user-card" data-user-id="123">
    <div data-user-card-target="cardContainer">
       <%= render UserCardComponent.new(user: User.find(123)) %>
    </div>
  <button data-action="user-card#refreshCard">Refresh Card</button>
</div>
```

In this scenario, when the button is clicked, `refreshCard()` in the `user_card_controller.js` makes a request to the server at `users/123/card` for a new version of the `UserCardComponent`, and replaces the inner html of the  `cardContainerTarget` element with this rendered component.

**Example 2: Component Updates with Data Payloads**

Now, consider a scenario where you need to partially update a component by passing data to it. This usually needs a more complex controller setup. Let’s say we want to change only the email displayed.

First, modify the component:

```ruby
# app/components/user_card_component.rb
  def initialize(user:, email_override: nil)
    @user = user
    @email_override = email_override
  end

  def user_email
    @email_override || @user.email
  end

  def template
   <<~HTML
      <div class="user-card">
        <h3>User: #{@user.name}</h3>
        <p>Email: #{user_email}</p>
      </div>
    HTML
  end
```

The controller now accepts a parameter:

```ruby
# app/controllers/users_controller.rb
  def card
    @user = User.find(params[:id])
    email_override = params[:email_override]
    render partial: "users/card", locals: { user: @user, email_override: email_override}
  end
```

And the Stimulus controller to send the request with a data payload:

```javascript
// app/javascript/controllers/user_card_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = [ "cardContainer" ]

  updateEmail() {
    const userId = this.element.dataset.userId;
    const newEmail = "updated.email@example.com" // For testing, ideally this comes from a form or input.

    fetch(`/users/${userId}/card?email_override=${encodeURIComponent(newEmail)}`)
      .then(response => response.text())
      .then(html => {
        this.cardContainerTarget.innerHTML = html;
      });
    }
}
```

and the HTML:

```html
<div data-controller="user-card" data-user-id="123">
    <div data-user-card-target="cardContainer">
        <%= render UserCardComponent.new(user: User.find(123)) %>
    </div>
  <button data-action="user-card#updateEmail">Update Email</button>
</div>
```

Now, pressing the button makes a request with an `email_override` parameter. This results in the `UserCardComponent` rendering with the changed email value.

**Example 3: Rendering a new component entirely**

Let's say we are loading an entirely different component based on some condition. We can reuse the approach but need more logic:

Component:

```ruby
# app/components/conditional_component.rb
class ConditionalComponent < ViewComponent::Base
  def initialize(condition:)
    @condition = condition
  end

  def template
    if @condition
      <<~HTML
       <div class="conditional-component">
        <p>Condition is true</p>
       </div>
     HTML
    else
     <<~HTML
       <div class="conditional-component">
          <p>Condition is false</p>
       </div>
     HTML
    end
  end
end
```

The controller:

```ruby
# app/controllers/conditional_controller.rb
def show
    condition = params[:condition] == 'true'
    render partial: "conditional/component", locals: { condition: condition }
  end
#app/views/conditional/_component.html.erb
<%= render ConditionalComponent.new(condition: condition) %>
```

Stimulus controller:

```javascript
// app/javascript/controllers/conditional_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = [ "container" ]

  changeComponent() {
      const condition = this.element.dataset.condition === 'true' ? 'false' : 'true';
       fetch(`/conditional/show?condition=${condition}`)
      .then(response => response.text())
      .then(html => {
        this.containerTarget.innerHTML = html;
        this.element.dataset.condition = condition
      });
  }
}
```

and the HTML with Stimulus bindings:
```html
<div data-controller="conditional" data-condition="true">
    <div data-conditional-target="container">
        <%= render ConditionalComponent.new(condition: true) %>
    </div>
    <button data-action="conditional#changeComponent">Change</button>
</div>
```

When the button is pressed the `changeComponent` function toggles the condition and fetches a new component.

In each of these examples, we use `fetch` to obtain the pre-rendered html from the server based on different parameters. This strategy isolates server-side rendering with `view_component` from client-side manipulation using Stimulus, making the code easier to reason about and to maintain.

For deeper learning on these techniques, I'd highly recommend reading *Rails 7 in Action* by Matt Swanson and *Programming Phoenix LiveView* by Bruce Tate, Sophie DeBenedetto, and Chris McCord. They'll provide a foundational and advanced understanding of the principles behind server-side rendering and managing dynamic user interfaces. Additionally, looking at the official documentation for both `view_component` and Stimulus will clarify the intricacies of each tool individually.

Remember, the key to success with these approaches is to meticulously separate concerns. Let view components handle rendering, and let Stimulus controllers handle user interactions and dom updates. This makes for more sustainable and scalable applications over time.
