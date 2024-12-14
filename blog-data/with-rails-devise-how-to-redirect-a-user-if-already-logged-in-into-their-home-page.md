---
title: "With Rails Devise, how to redirect a user if already logged in, into their home page?"
date: "2024-12-14"
id: "with-rails-devise-how-to-redirect-a-user-if-already-logged-in-into-their-home-page"
---

alright, so you're having that classic devise redirect dance, where a user, already logged in, tries to hit the sign-in page and instead of being bumped to their actual home page, they're stuck looking at the login form again. i've definitely been there, spent a few late nights staring at rails logs chasing this one down. it's a pretty common issue when getting started with devise, and honestly even later when you've got more complex setups.

the core problem here is that devise, by default, assumes if a user is visiting the sign-in path, they want to sign in. it doesn't inherently know, "hey, this person is already good to go, send them elsewhere". so, we need to inject some logic to tell devise to be a bit more clever.

the simplest solution often involves tweaking the devise controller. specifically, you need to modify the `sessions#new` action (which is the one that renders the sign-in form). normally this method would just render the form, we need to add a check for current user and redirect if already signed in. this is a pretty common pattern when using devise.

let's start with the standard setup. assume your routes are like:

```ruby
devise_for :users
root to: 'home#index'
```

and you have a `home_controller.rb` with an `index` action.

```ruby
class HomeController < ApplicationController
  def index
    # your homepage logic here
    render plain: "welcome home user!"
  end
end
```

now, you want to stop a logged in user going to `/users/sign_in` and instead redirect to `/`. first create a file at `app/controllers/users/sessions_controller.rb` that looks like this:

```ruby
class Users::SessionsController < Devise::SessionsController

  def new
    if user_signed_in?
      redirect_to root_path, notice: 'already logged in'
    else
      super
    end
  end
end
```

next tell devise to use this custom controller, add to the `devise_for :users` in `config/routes.rb` the parameter `controllers: { sessions: 'users/sessions' }` like below:

```ruby
devise_for :users, controllers: { sessions: 'users/sessions' }
root to: 'home#index'
```

what's happening in the code? we're overriding the `new` method. if `user_signed_in?` is true, we send them to the root path using `redirect_to root_path`. otherwise we call `super`, which will be the standard devise `new` method that renders the sign in form. the `:notice` gives the user a message that shows up on the root path, which is optional, and can be removed if you do not want to show a notice.

this handles the most common case but sometimes your application has more complex authorization schemes. maybe you're using roles or different home pages based on user type. i remember one project, where we had admin, customer and partner users and each of them had a different dashboard that they would go to. this can be achieved by modifying the after_sign_in_path_for method in a similar way.

we'll need to override the `after_sign_in_path_for` method, this method is called by devise right after the user is successfully logged in.

create a file at `app/controllers/application_controller.rb` and if it already exists, just add this code to the file:

```ruby
class ApplicationController < ActionController::Base
    protect_from_forgery with: :exception

    def after_sign_in_path_for(resource)
      if resource.is_a?(User)
          if resource.admin?
            admin_dashboard_path
          elsif resource.partner?
            partner_dashboard_path
          else
            root_path
          end
      else
        super
      end
    end
end
```

let me break down what is going on. we are checking if the `resource` param passed to the method is a `User`, we are then checking the role of the user, and according to it we redirect it to a different path. you have to create the routes and controllers for `admin_dashboard_path` and `partner_dashboard_path` and change the code according to your needs.

the `admin?` and `partner?` methods must be defined in the `User` model, you can use enums or simple booleans in the `users` table.

we are also handling cases where the resource is not a `User` model. this happens when you have multiple devise resources, like `admin` for example. in this case we are just calling `super` which defaults to the standard devise behavior.

last, but not least there is the case that a user is already signed in and the want to go to `/users/sign_up`. in this case, the redirect has to be handled in the `registrations#new` action. let's create a custom controller for this. create a file at `app/controllers/users/registrations_controller.rb`:

```ruby
class Users::RegistrationsController < Devise::RegistrationsController

  def new
    if user_signed_in?
      redirect_to root_path, notice: 'already logged in'
    else
      super
    end
  end
end
```

next tell devise to use this custom controller, add to the `devise_for :users` in `config/routes.rb` the parameter `controllers: { sessions: 'users/sessions', registrations: 'users/registrations' }` like below:

```ruby
devise_for :users, controllers: { sessions: 'users/sessions', registrations: 'users/registrations' }
root to: 'home#index'
```

this controller is almost identical to the sessions controller, the difference is that this code handles cases when the user wants to go to `/users/sign_up` route.

now this should take care of most standard cases, but there is a myriad of other possibilities to handle these redirects. this are some basic pointers. this type of redirects when already signed in is really common and should be handled by each and every web application that uses user authentication.

if you want to really understand all the edge cases, i would recommend having a deep dive into the [rails guides on routing](https://guides.rubyonrails.org/routing.html), and also the [devise documentation](https://github.com/heartcombo/devise). there are also some chapters in [the "agile web development with rails" book](https://pragprog.com/titles/rails7/agile-web-development-with-rails-7/) that cover devise in detail and go deep into custom authentication schemes.

it also worth noticing that if you are developing an api it would be better to respond with the apropriate http status code instead of a redirect. but that's another can of worms for another day, my advice would be to keep it as simple as possible, because code is like a joke, if you have to explain it then it’s bad.
