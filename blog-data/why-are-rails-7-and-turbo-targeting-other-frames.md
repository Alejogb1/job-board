---
title: "Why are Rails 7 and Turbo targeting other frames?"
date: "2024-12-14"
id: "why-are-rails-7-and-turbo-targeting-other-frames"
---

well, this is a question i've bumped into a bunch, and it's one that often gets folks scratching their heads when they're moving from older ways of doing things in rails. it’s like, you think you’ve got a handle on how rails handles updates, then turbo comes along and decides to talk to different parts of your page entirely. i've been there, staring at the console, wondering why my carefully crafted form update was refreshing the whole darn thing instead of just a tiny little area. so, let's break it down, how it works and why it works that way, based on my own experiences dealing with this kind of thing, and without a lot of overly flowery language.

first things first, we have to understand the core shift in thinking that turbo introduces. before turbo, the usual rails way of updating content was with a full page refresh, or sometimes using javascript like ajax requests which you would use in order to manually update specific parts of the page after receiving a json response from the server. but, turbo's approach is different, it's all about partial page updates. it wants to only change the elements on the page that actually *need* to be changed, this is done through something called turbo frames.

turbo frames, they are pretty much the bread and butter of this whole thing. think of them like little containers on your page. each frame is identified with an id, and when a turbo request happens, rails looks for a matching frame on the page, based on the id. if it finds one, it replaces only the content within that frame, this drastically reduces the amount of data your browser needs to download and process, leading to a much faster, more responsive user experience.

so why target different frames? the reason is because of these partial updates. instead of being limited to updating the entire page after every action, you can now have multiple areas of your page updating independently based on user actions, this allows you to have more sophisticated interactions without suffering a performance hit. say you’ve got a product page with a cart in the sidebar and you also have a main product area. before, updating the cart after adding a product would mean a full reload, including loading the images and the product description again, and that would make the page reload sluggish and cumbersome. with turbo frames, you can have each of those parts in a frame, so updating the cart will refresh the cart frame, but not the product information. this kind of granularity is what makes turbo powerful.

let me give you a taste of this using my own past errors. when i was first learning this, i was building a simple todo list application. i had a form to add a new todo item, and then the list of todo items displayed below. my initial setup had both the form and the list of todos within the same frame, let's call it `todo_frame`. when i submitted a new todo, everything in `todo_frame` got updated, which is what i expected, so that made sense. my code back then looked something similar to this example:

```erb
<div id="todo_frame">
  <%= form_with(url: todos_path, method: :post, data: { turbo_frame: 'todo_frame' }) do |form| %>
    <%= form.text_field :description %>
    <%= form.submit "add todo" %>
  <% end %>

  <ul id="todos">
    <%= render @todos %>
  </ul>
</div>
```
```ruby
# controller
def create
  @todo = Todo.create(params.require(:todo).permit(:description))
  @todos = Todo.all

  respond_to do |format|
    format.turbo_stream { render turbo_stream: turbo_stream.replace('todo_frame', partial: 'todos/todo_list', locals: { todos: @todos}) }
    format.html { redirect_to todos_path }
  end
end
```
```erb
# partial (todos/_todo_list.html.erb)
<div id="todo_frame">
  <ul id="todos">
    <%= render @todos %>
  </ul>
</div>
```

the code was working, it was doing what it was supposed to do, but it was a bit inefficient, it was re rendering the entire frame which had the form inside of it as well, when the form was not changing at all, and i was going to notice that especially with more complex forms. it was working as expected at that moment, but when the project got bigger and the form got more complex, this became a performance bottleneck. because i was replacing the entire frame with the list and the form. i eventually realized that this was suboptimal and unnecessary, it was reloading the entire `todo_frame`, even though i was only updating the list itself.

then i realized that i needed to split things out into distinct frames, so i decided to make a separate frame just for the list. so the form was not included inside the frame anymore, so my form would be static, and when i would submit a new todo item, only the list part would be re-rendered. the updated code ended looking like this:

```erb
<%= form_with(url: todos_path, method: :post, data: { turbo_frame: 'todos' }) do |form| %>
    <%= form.text_field :description %>
    <%= form.submit "add todo" %>
  <% end %>

<div id="todos">
  <ul id="todos_list">
    <%= render @todos %>
  </ul>
</div>
```
```ruby
# controller
def create
  @todo = Todo.create(params.require(:todo).permit(:description))
  @todos = Todo.all
  
  respond_to do |format|
    format.turbo_stream { render turbo_stream: turbo_stream.replace('todos', partial: 'todos/todo_list', locals: { todos: @todos}) }
    format.html { redirect_to todos_path }
  end
end
```

```erb
# partial (todos/_todo_list.html.erb)
<ul id="todos_list">
    <%= render @todos %>
</ul>
```

now, when a new todo is added, the controller would only return the html of the todo list. the form would not be affected, only the list gets updated. it was much more performant and more elegant than the first iteration, and it improved responsiveness greatly.

this example illustrates why targeting different frames is useful: it makes it possible to update only the parts of your page that have changed after some event. imagine how much faster things become when you only have to update the list of todos and not re-render the form, it's like giving your webpage a shot of espresso, it becomes much more responsive. and this is exactly what turbo’s philosophy is, less work for the server and less work for the browser.

this approach is also closely tied with the concept of *hotwire*, it's the idea of building rich, single page applications without using a lot of javascript. the bulk of the logic stays on the server side, and turbo handles the updates on the front-end, using html over the wire, there is a minimal amount of javascript needed to make all of this work and it's usually handled by the turbo library itself. it's also beneficial if you are trying to maintain applications with legacy front end code. it makes the front-end code much simpler and easier to maintain, and by moving most of the complex logic to the server side, you can focus on writing clean, efficient code.

and if i can get a little philosophical here, this approach forces you to think about your page in a more component-oriented way. you start dividing things into self-contained, independently updatable pieces, and this often leads to a cleaner and more modular codebase. it's kind of like organizing your messy desk, once you’ve got everything in its right place, you will be more efficient at finding stuff.

if you are looking for good materials to learn more about this, i would highly recommend looking at the *hotwire handbook*, it is a fantastic resource for all things hotwire and turbo, it will go deeper in the mechanics and the philosophy behind it, and will really solidify the concepts. and for more hands-on experience, the *rails guides* section on turbo is also great. it provides real-world examples and demonstrates how to integrate turbo seamlessly into your rails applications. there is also a great section on *turbo stream*, which is how rails communicates with the browser to do partial page updates, this resource will show the most efficient patterns for updating different elements.

so, that’s my take on why rails 7 and turbo target different frames. it's all about creating more efficient and responsive web applications, and the ability to have granular control over page updates is a game changer, and while it might take some time to fully grasp the concepts at first, once you understand the power behind it, i guarantee you will never want to go back to the old ways.

ps: and i also have the feeling that sometimes frameworks change just for the sake of changing things, you know what i mean?
