---
title: "With Ruby on Rails (Redmine), how to add a route?"
date: "2024-12-15"
id: "with-ruby-on-rails-redmine-how-to-add-a-route"
---

alright, so you're looking at adding a route in your redmine rails app, huh? been there, done that, got the t-shirt... and probably a few bug reports filed against my name too. it's one of those things that seems simple on the surface, but can get a little twisty depending on what you're trying to achieve. let's break it down.

first off, we need to remember that routing in rails is all about mapping urls to controller actions. basically, when someone types a url into their browser, rails looks at its routing table to figure out which controller and method should handle the request.

in redmine, like most rails apps, these routes are defined in the `config/routes.rb` file. this is where the magic happens, or sometimes where the hair-pulling starts. so, i assume you already know where this file resides. if not, well, now you do. it's like the central nervous system of your url handling. messing this up can really make things go south, trust me. i once spent a whole weekend trying to figure out why a particular page kept 404'ing, turned out i had a typo in a single line in this very file. those were dark times, let me tell you. i've been there, learned my lesson. hopefully you can learn from my past mistakes without having to repeat them.

the core concept is that you'll be using methods like `get`, `post`, `put`, `patch`, and `delete` to define your routes. these correspond to the http verbs and tell rails what to do based on the type of request coming in. for example, `get` is typically used for fetching data, while `post` is for creating new data.

a simple example would be if you wanted to add a route to view a custom "report" within redmine. imagine you have a controller called `reports_controller.rb` and an action within that controller called `show`.

here's how the `routes.rb` file might look:

```ruby
  Rails.application.routes.draw do
    resources :projects do
      get 'reports/:report_id', to: 'reports#show', as: 'project_report'
      #other project routes
    end
    #other routes ...
  end
```

this snippet does the following: it adds the route `projects/:project_id/reports/:report_id` to your redmine instance. it will respond to get request, send it to the reports controller's show method and name the route `project_report` allowing you to use the `project_report_path` helper to build the url programatically elsewhere in the app. using resources also gives you access to helper path such as `edit_project_path` or `project_path`. always a good idea to use rails helpers for routing.

so, to call this route you could go to something like `/projects/123/reports/456` assuming your `project_id` is `123` and the `report_id` is `456`. the `reports#show` bit means it should be handled by the `show` action in `reports_controller.rb`. this is called a path segment and the value of `:report_id` will be available via params as `params[:report_id]` and the `params[:project_id]` too of course.

now, if you wanted to create a new report, you might add a route that uses the `post` verb:

```ruby
Rails.application.routes.draw do
  resources :projects do
    get 'reports/:report_id', to: 'reports#show', as: 'project_report'
    post 'reports', to: 'reports#create', as: 'project_reports'
    #other project routes
  end
  #other routes ...
end
```

here, we've added `post 'reports', to: 'reports#create', as: 'project_reports'`. this means that when a form is submitted with the post method to `/projects/123/reports` url, it will go to the create method inside the reports controller. the url will be available with `project_reports_path` rails helper.

another thing you might do is adding custom named routes. for example if you want a specific route to add a user to a project you would use a `member` block of code:

```ruby
Rails.application.routes.draw do
  resources :projects do
    get 'reports/:report_id', to: 'reports#show', as: 'project_report'
    post 'reports', to: 'reports#create', as: 'project_reports'
    member do
      post 'add_user/:user_id', to: 'projects#add_user', as: 'project_add_user'
    end
  #other project routes
  end
  #other routes ...
end
```

this gives you a route to call `projects#add_user` passing both params `:project_id` and `:user_id` and gives you access to the `project_add_user_path` helper. and you would call it with something like `projects/123/add_user/789` in this example.

remember that the order of routes matters, rails processes these from top to bottom. so, if you have conflicting routes, the first one that matches will be used. i once had a problem where all requests went to the first defined route because i had put a wildcard route that was too generic and swallowed the following ones. it was my "facepalm moment" of the week. debugging can be a time sinker sometimes.

as a best practice i recommend structuring your routes in a clear and logical way and avoiding adding unnecessary complexity. using the `resources` method where it makes sense, along with the `member` or `collection` method blocks to add custom logic will save you time later. using named routes is a good practice, it makes using the urls programatically much easier.

to help further with your quest, if you're looking for deeper explanations, i'd recommend reading "agile web development with rails 6". it has a very good section on routing that goes into great detail. another useful book is "the rails 7 way" if you are using a newer version. also, the official rails documentation on routing is a treasure trove of information. i strongly suggest you have a look at it. it will definitely save you time and pain. i keep the documentation pages open almost all of the time. there is no shame in reading the docs.

that's about it, i think i've covered most of the key aspects. if you get stuck, remember to double check your `routes.rb` file for typos and ensure the path, verb, and controller/action are as you expect. and the joke? why don't scientists trust atoms? because they make up everything!

happy coding! let me know if anything else pops up, i'm here to help, or at least to commiserate.
