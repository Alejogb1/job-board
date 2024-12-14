---
title: "How can I pass a variable from X_datatable.rb to view?"
date: "2024-12-14"
id: "how-can-i-pass-a-variable-from-xdatatablerb-to-view"
---

so, you're wanting to get data from a ruby datatable class over to your view, that's a pretty common scenario i've bumped into a bunch. let's talk about how i usually handle this, and some of the things i've learned from making mistakes along the way.

first up, let's assume you've got something like this in your ruby file, let's call it `x_datatable.rb`:

```ruby
# x_datatable.rb
class XDatatable
  def initialize(data_source)
    @data_source = data_source
  end

  def data
    # pretend this is a complex query or operation to get data
    @data_source.map { |item| { id: item[:id], name: item[:name].upcase, created_at: item[:created_at].strftime('%Y-%m-%d') } }
  end

end
```

this `x_datatable` class is doing some basic data processing. in a real-world application this could be much more intricate, involving complex database queries, calculations, whatever. i've had datatables that would fetch data from multiple databases, filter, paginate, and even format dates differently based on the user's location. it's all about that specific need of yours. anyway, we have this class and now need the data in a view, like an erb file.

the simple way to get this over to the view is usually through a controller. it acts as the bridge.

here's an example of what that might look like in a controller, maybe `items_controller.rb`:

```ruby
# items_controller.rb
class ItemsController < ApplicationController
  def index
    data_source = [
       { id: 1, name: 'item one', created_at: Time.now - 1.day },
       { id: 2, name: 'item two', created_at: Time.now },
       { id: 3, name: 'item three', created_at: Time.now + 1.day }
      ]
    datatable = XDatatable.new(data_source)
    @items = datatable.data # data is now in an instance variable
  end
end

```

what's happening here:

1.  we instantiate `xdatatable` using some example data; in your real scenario, you'd probably get this data from the database or other source.
2.  we call the `data` method on the datatable instance.
3.  the result is stored into an instance variable called `@items`.

note that in the controller i'm calling the `data` method, that's a very usual way but there are others. some people call it `to_a` or `as_json` or something that implies that is extracting the data. the actual name i feel is not relevant as long the intention is clear.

now that we have the `@items` instance variable it's available in your erb template, like `index.html.erb`:

```html+erb
<%# index.html.erb %>
<table>
  <thead>
    <tr>
      <th>id</th>
      <th>name</th>
      <th>created at</th>
    </tr>
  </thead>
  <tbody>
    <% @items.each do |item| %>
      <tr>
        <td><%= item[:id] %></td>
        <td><%= item[:name] %></td>
        <td><%= item[:created_at] %></td>
      </tr>
    <% end %>
  </tbody>
</table>

```

this is the typical way you see a loop through a data set in a rails application.

a key part here is that instance variables in rails controllers, starting with `@`, are automatically passed to the view. that's how the erb template sees `@items`.  i remember one time i forgot the `@` and the view was completely blank, took me a while to understand what was happening, i was assigning the data to a local variable inside the controller, not an instance one. i spent so much time searching online trying to find the issue.

now, let's touch on a few common pitfalls, things i've seen folks mess up, myself included.

*   **avoiding complex logic in the view:** your view's main job is to display data. you want to avoid doing actual data transformations here if possible. move the complex logic to the datatable or controller (sometimes helper methods, but those are less usual for data manipulation). i've seen views with so much logic inside, it became a nightmare to maintain. moving this to the datatable cleans things up a lot.

*   **performance:** if the dataset is very large, you'll need to consider pagination, eager loading, and other optimizations. loading all data in a single shot may end up bringing your application to a halt when you have a lot of rows. i’ve crashed a couple of servers learning this lesson. one thing is loading small datasets another one is when you start talking about hundreds of thousands of rows, you need to think more carefully.

*   **testing:** make sure to test your datatable class independently of the controller. the controller usually is just an orchestrator to gather information and pass it to the view. the core of the data transformation logic should be in the datatable class because it's easier to test there. i've seen controllers that had way too much code, became really difficult to test. having the core logic inside a class that you can instantiate and test independently is way better.

regarding resources, while there's not one single book on *exactly* this, you'll find that "agile web development with rails 7" by sam ruby is a good, broad book that will help you to build web applications correctly, it goes over all of these subjects. as a more specific resource, check out "refactoring: improving the design of existing code" by martin fowler; it is not directly related to rails but it will help you in making better classes, and designing your data and transformation layers better.

one joke? how does a ruby developer organize their spices? in arrays, obviously!

to summarize, the way i normally deal with this issue is: the data transformations happen inside the datatable class, the controller gets the data from the datatable, and passes it to the view using an instance variable. keep the view simple, avoid putting business logic there. think about performance, and test, test, test. this approach should cover most common cases. let me know if you have more specific situations.
