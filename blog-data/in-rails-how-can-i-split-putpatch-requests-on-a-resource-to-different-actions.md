---
title: "In Rails, how can I split PUT/PATCH requests on a resource to different actions?"
date: "2024-12-15"
id: "in-rails-how-can-i-split-putpatch-requests-on-a-resource-to-different-actions"
---

alright, so you're looking at how to route put/patch requests in rails to different controller actions, huh? been there, done that, got the t-shirt – and several bug reports. it's a common enough need, especially when your api starts getting more nuanced.

i remember way back when, i was working on this crud app for managing, i guess you could call them, "widgets." initially, it was straightforward: a put or patch to `/widgets/1` would just update the whole widget, everything all at once, a big messy blob of json being passed. but as the widget concept evolved, we needed to be able to update different aspects of it separately. one update for its basic info, another for its, let's say "advanced settings," and yet another for its "state." that meant we couldn't just rely on the standard rails restful routing for put and patch anymore. that's where i learned some hard lessons about resource routing customization, and thankfully some rails magic as well.

the default rails routing, as you might already be aware, is pretty rigid when it comes to put/patch. it generally points any put or patch request to a single `update` action in your controller. but rails, being rails, gives us hooks to change that. the simplest method, and where i started, is adding custom routes.

basically, instead of having rails auto-magically infer everything, you explicitly tell it "if you see this path with this verb, go to *this* action." here's how that looks in your `routes.rb` file:

```ruby
resources :widgets do
  member do
    put 'update_info', to: 'widgets#update_info'
    patch 'update_settings', to: 'widgets#update_settings'
    put 'update_state', to: 'widgets#update_state'
  end
end
```

this setup here, it creates three different endpoints for you under `/widgets/1`: `/widgets/1/update_info`, `/widgets/1/update_settings`, and `/widgets/1/update_state`. all of them using either put or patch verbs. each of these now points to its own distinct action in the widgets controller. the `member` block tells rails that these routes pertain to a *specific* widget instance, not the whole collection.

inside the controller, you'd have corresponding action methods:

```ruby
class WidgetsController < ApplicationController
  def update_info
    widget = Widget.find(params[:id])
    if widget.update(params.require(:widget).permit(:name, :description))
      render json: widget, status: :ok
    else
     render json: widget.errors, status: :unprocessable_entity
    end
  end

  def update_settings
     widget = Widget.find(params[:id])
     if widget.update(params.require(:widget).permit(:setting1, :setting2))
        render json: widget, status: :ok
     else
        render json: widget.errors, status: :unprocessable_entity
     end
  end

  def update_state
     widget = Widget.find(params[:id])
     if widget.update(params.require(:widget).permit(:state))
        render json: widget, status: :ok
     else
        render json: widget.errors, status: :unprocessable_entity
     end
  end
end
```

now each method handles a different type of update to the widget. the important part here is the `params.require(:widget).permit(...)` which allows you to whitelist what parameters can be used for each type of update request ensuring that requests only update what is meant to be updated in each action. notice that you can choose any specific verb on each of the route definitions, not just "put" or "patch".

this approach is fairly straightforward and it's how i handled most of my needs early on. but as time went by, our "widgets" app got a bit more complex (don't they always?) and we wanted to get rid of all the additional segments on our endpoints paths. instead of having `/widgets/1/update_info`, we wanted to keep it as `/widgets/1` but have different request payloads going to their own actions.

that's where i learned the power of custom routing constraints. constraints allow you to route a request based on attributes of the request itself, like the headers or the body content. we used the request content-type as a parameter of our constraint.

```ruby
  put '/widgets/:id', to: 'widgets#update', constraints: { content_type: /application\/vnd\.widget\.info\+json/ }
  patch '/widgets/:id', to: 'widgets#update', constraints: { content_type: /application\/vnd\.widget\.settings\+json/ }
  put '/widgets/:id', to: 'widgets#update', constraints: { content_type: /application\/vnd\.widget\.state\+json/ }
  resources :widgets
```
here, we are intercepting the put and patch verbs to the `/widgets/:id` endpoint and based on the request header `content-type` we are routing to the `update` action with the different content type constraint. and we keep the default `resources :widgets` so it handles other requests such as `get /widgets` or `post /widgets` as the standard restful methods. the controller would now look like this:

```ruby
class WidgetsController < ApplicationController
  def update
    widget = Widget.find(params[:id])

    case request.content_type
    when 'application/vnd.widget.info+json'
      if widget.update(params.require(:widget).permit(:name, :description))
        render json: widget, status: :ok
      else
         render json: widget.errors, status: :unprocessable_entity
      end
    when 'application/vnd.widget.settings+json'
      if widget.update(params.require(:widget).permit(:setting1, :setting2))
        render json: widget, status: :ok
      else
         render json: widget.errors, status: :unprocessable_entity
      end
    when 'application/vnd.widget.state+json'
       if widget.update(params.require(:widget).permit(:state))
          render json: widget, status: :ok
       else
          render json: widget.errors, status: :unprocessable_entity
       end
    else
      head :bad_request # this means no handler for this request content type
    end
  end
end
```

now, the same `/widgets/1` endpoint will update the widget differently depending on the content type you send in the request. each different content type allows a different set of parameters to be received.

for me this has been a useful approach for creating versioned restful endpoints without altering the rest standard, or having additional segments in the endpoint path. if you go with this path remember to properly document your api.

i've also used similar tactics with request headers other than content type to have more control, but content type is the usual pattern. i've even seen people using json structures within the body to determine how to route requests. if you go that path it will look something like this (example only, do not use it with the same content type example we had before):

```ruby
  put '/widgets/:id', to: 'widgets#update', constraints: lambda { |request| request.params['update_type'] == 'info' }
  put '/widgets/:id', to: 'widgets#update', constraints: lambda { |request| request.params['update_type'] == 'settings' }
  put '/widgets/:id', to: 'widgets#update', constraints: lambda { |request| request.params['update_type'] == 'state' }
  resources :widgets
```
and the controller:
```ruby
class WidgetsController < ApplicationController
  def update
    widget = Widget.find(params[:id])

    case params['update_type']
    when 'info'
      if widget.update(params.require(:widget).permit(:name, :description))
        render json: widget, status: :ok
      else
        render json: widget.errors, status: :unprocessable_entity
      end
    when 'settings'
      if widget.update(params.require(:widget).permit(:setting1, :setting2))
        render json: widget, status: :ok
      else
        render json: widget.errors, status: :unprocessable_entity
      end
    when 'state'
      if widget.update(params.require(:widget).permit(:state))
        render json: widget, status: :ok
      else
         render json: widget.errors, status: :unprocessable_entity
      end
    else
      head :bad_request
    end
  end
end
```
this last one uses the content of the request itself using lambda on the routing constraint. it works but i prefer the content type constraint, it just feels cleaner.

remember the order in which these routes are defined in your `routes.rb` file matters, rails checks routes in the order they appear. so, if you define the generic `resources :widgets` *before* you define your custom constrained routes, rails will hit the generic route first and your custom routing will never fire.

this was what i went through when i started splitting my put/patch requests for crud resource actions. and don't worry we have all been there trying to figure this out. the key is to be patient and break things down. you can get it working, and once you do it's a valuable tool to add to your api dev arsenal. also, i once had a system that worked like a charm, and then we added a new server, and poof, everything broke. i learned more about load balancers that day than i ever wanted to. yeah, it was a fun monday!

as for where to go from here, for the concepts of routing constraints and more details on rails routes, i would advise checking the official rails guides, specifically the section on "rails routing from the outside in". also, the "crafting rails applications" book by jose valim gives a deeper dive in routing that goes beyond the basics, i found it quite useful.
