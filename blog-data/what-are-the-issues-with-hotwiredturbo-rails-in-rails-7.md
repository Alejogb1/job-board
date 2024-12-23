---
title: "What are the issues with @hotwired/turbo-rails in Rails 7?"
date: "2024-12-23"
id: "what-are-the-issues-with-hotwiredturbo-rails-in-rails-7"
---

, let's talk about `@hotwired/turbo-rails` in Rails 7, specifically the challenges I've encountered and how I've addressed them. It's a powerful tool, no doubt, but like any technology, it presents its own set of interesting puzzles. I’ve spent a fair amount of time working with it, going back to when it was brand new, and I’ve got a few scars to show for it.

The core of Turbo's appeal lies in its ability to accelerate page interactions through selective DOM updates via websockets and server-rendered html fragments. However, the transition to this approach wasn’t always frictionless. Here’s a breakdown of what I've faced:

One of the primary challenges that I experienced early on was managing complex form interactions, particularly those involving nested attributes. Turbo's default behavior for form submissions can be, shall we say, *surprising* if you’re used to traditional Rails UJS. Specifically, when a form fails to validate, or requires a server-side redirect (for example, after creating a record), it can lead to a rather inconsistent user experience. The primary issue is that Turbo's partial updates might not accurately reflect the state of the form after processing. This can especially happen with validation errors. The typical behavior is for the server to render the same form with validation errors, and this needs to be gracefully handled by Turbo to avoid issues like losing user input or showing incorrect messages.

For instance, imagine a form for creating a user with an address. Initially, I found myself relying heavily on the default Turbo behavior which, when a validation error occurred, returned the form partial again. This sometimes resulted in a brief, disorienting flash of the unvalidated form, and worse, a state where error messages didn’t render reliably. What I eventually realized is that for forms like this, with nested attributes or complex interactions, a combination of techniques is often necessary. This involves careful use of the `turbo_stream` responses, custom javascript events, and, occasionally, a more declarative approach to rendering error messages in the HTML itself, instead of solely relying on Turbo’s replacement behavior.

Here’s a concrete example. Consider a user form with nested address attributes. Here's how the server-side handling would look when a validation fails, using `turbo_stream`:

```ruby
# users_controller.rb
class UsersController < ApplicationController
  def create
    @user = User.new(user_params)
    if @user.save
      redirect_to @user, notice: 'User created successfully.'
    else
      render turbo_stream: turbo_stream.replace('user_form', partial: 'form', locals: { user: @user })
    end
  end

  private

  def user_params
    params.require(:user).permit(:name, address_attributes: [:street, :city, :state])
  end
end
```

```erb
# _form.html.erb partial
<%= turbo_frame_tag 'user_form' do %>
  <%= form_with model: user, data: { turbo: false } do |form| %>
    <% if user.errors.any? %>
      <div id="error_explanation">
        <h2><%= pluralize(user.errors.count, "error") %> prohibited this user from being saved:</h2>

        <ul>
          <% user.errors.each do |error| %>
            <li><%= error.full_message %></li>
          <% end %>
        </ul>
      </div>
    <% end %>

    <div>
      <%= form.label :name, style: "display: block" %>
      <%= form.text_field :name %>
    </div>

    <div>
      <%= form.fields_for :address do |address_form| %>
        <div>
          <%= address_form.label :street, style: "display: block" %>
          <%= address_form.text_field :street %>
        </div>
        <div>
          <%= address_form.label :city, style: "display: block" %>
          <%= address_form.text_field :city %>
        </div>
        <div>
          <%= address_form.label :state, style: "display: block" %>
          <%= address_form.text_field :state %>
        </div>
      <% end %>
    </div>
    <div>
      <%= form.submit %>
    </div>
  <% end %>
<% end %>

```

In this example, the `turbo_stream.replace` call replaces the entire `user_form` tag with the re-rendered form, ensuring the error messages are correctly displayed. Importantly, setting `data: { turbo: false }` on the form disables Turbo processing for that particular submission, allowing standard form processing. This prevents potential unexpected behaviors with Turbo.

Another issue I initially ran into was managing more sophisticated UI interactions beyond basic form handling. I recall working on a project with a complex dashboard that relied heavily on modal windows. Initially, I tried to rely purely on Turbo's streaming capabilities to handle the modal content updates, which led to race conditions. Specifically, closing a modal in one location before the server responded with the new content, would cause parts of the page to not update correctly. The lesson learned here was the importance of combining Turbo with custom javascript when you needed more controlled and synchronous behavior for user interactions. For these situations, you can utilize custom javascript to manage the modal lifecycle more reliably, while still using Turbo for content updates. We could use dispatch custom javascript events, combined with Turbo streams to ensure smooth transitions.

Here’s an example demonstrating how to leverage Turbo streams alongside custom JavaScript to control the modal lifecycle:

```javascript
// application.js
document.addEventListener('turbo:load', function() {
  document.addEventListener('click', function(event) {
    if (event.target.matches('[data-modal-trigger]')) {
      event.preventDefault();
      const target = event.target.getAttribute('data-modal-target');
      document.getElementById(target).classList.add('modal--open');

        // Ensure that Turbo is aware of these changes and avoids reloads that will
        // discard our open modal
    }

    if (event.target.matches('[data-modal-close]')) {
      event.preventDefault();
       const modal = event.target.closest('.modal');
        modal.classList.remove('modal--open');
    }
  });
});
```

```erb
# _modal.html.erb partial
  <div id="<%= modal_id %>" class="modal">
    <div class="modal-content">
       <span data-modal-close>&times;</span>
         <%= yield %>
    </div>
  </div>

```
```erb
# some_view.html.erb
<%= button_tag "Open Modal", data: { modal_trigger: true, modal_target: "my-modal" }  %>
<%= render 'modal', modal_id: "my-modal" do %>
 <p>This is a Modal content!</p>
 <%= turbo_frame_tag 'update_modal_content' do %>
   <%= link_to "Update Modal", update_path,  data: { turbo_stream: true } %>
 <% end %>
<% end %>
```

```ruby
# some_controller.rb
def update
    respond_to do |format|
      format.turbo_stream do
        render turbo_stream: turbo_stream.replace('update_modal_content', partial: 'new_modal_content')
      end
    end
  end

```

In this scenario, the `modal--open` css class handles toggling the display of the modal. The event listeners in `application.js` capture the opening and closing of the modal via data attributes. Then a standard `turbo_stream` action is used to update the content. This isolates the modal opening/closing from the content update, making it less susceptible to race conditions and timing issues.

Finally, dealing with server-side events pushed by websockets (using ActionCable) in conjunction with Turbo can be tricky. You have to carefully think about how to integrate those server-side pushes with Turbo streams, especially when those updates are meant to affect multiple connected clients. Often I found that you need to make sure the client explicitly knows it has been updated to avoid confusing the user about which is the most recent state.

For example, if you were creating a real-time chat application, it’s vital to ensure messages pushed through websockets are rendered seamlessly with Turbo's partial update mechanism, rather than having the user experience a clunky full page reload.

Here’s a simplified example of integrating websockets with Turbo streams:

```ruby
# channels/chat_channel.rb
class ChatChannel < ApplicationCable::Channel
  def subscribed
    stream_from "chat_channel"
  end

  def receive(data)
    message = Message.create!(content: data['message'])
    ActionCable.server.broadcast('chat_channel', {
                                 message: render_to_string(partial: 'messages/message', locals: { message: message }),
                                  turbo_stream: true
                                })
  end
end
```
```erb
# messages/_message.html.erb
 <div class="message">
    <p><%= message.content %></p>
 </div>
```
```javascript
// application.js
import { Turbo } from "@hotwired/turbo-rails"
import * as ActionCable from "@rails/actioncable"
window.addEventListener('turbo:load', () => {
     const cable =  ActionCable.createConsumer()
     cable.subscriptions.create('ChatChannel', {
            connected() {
                // Called when the subscription is ready for use on the server
            },

            disconnected() {
                // Called when the subscription has been terminated by the server
            },

            received(data) {
              if(data.turbo_stream){
                   Turbo.renderStreamMessage(data.message)
              }
           },
           send_message(message) {
             this.perform("receive", { message: message })
           }

     })
     document.getElementById("send_message").addEventListener("click", function(e) {
       e.preventDefault()
       const messageInput = document.getElementById("message_input")
       cable.subscriptions.subscriptions[0].send_message(messageInput.value)
       messageInput.value = "";
     })

    })
```

```erb
# some_view.html.erb
  <input id="message_input" type="text" />
  <button id="send_message">Send</button>
  <div id="message_container">
  </div>
```

In this example, we use the `ActionCable` javascript client to receive server sent messages via the web socket, and trigger a Turbo stream update via `Turbo.renderStreamMessage`. The server is broadcasting a render stream message which the client javascript code then applies to the dom.

In essence, while `@hotwired/turbo-rails` offers substantial improvements in user experience, it requires thoughtful implementation, especially when dealing with complex interactions. As you can see, I’ve had to combine Turbo’s functionality with a bit of custom Javascript and precise server-side responses to work through issues. For deeper dives, I recommend the official Hotwire documentation, the book "Programming Phoenix LiveView" by Bruce Tate, which has many principles that are applicable to Hotwire, and “Refactoring” by Martin Fowler, for general patterns that can be applied to improve code clarity. These resources will help you to not only grasp the fundamentals but also navigate the nuances of this powerful, but sometimes challenging, framework.
