---
title: "How to setup individual Content-Security_Policy for grape-swagger-rails?"
date: "2024-12-15"
id: "how-to-setup-individual-content-securitypolicy-for-grape-swagger-rails"
---

alright, so you're tackling content security policy (csp) with grape-swagger-rails, a classic situation. i've been down this road myself, more times than i'd care to remember. it always starts with that initial "oh, this should be straightforward" thought, and then...well, then the nuances hit you. especially when you're trying to juggle multiple parts of an application and making sure each of them doesn't accidentally break the others.

let's get into it. grape-swagger-rails, if you've worked with it before, is pretty good at generating swagger documentation but doesn't natively offer much control over individual csp headers. by default, it tends to use whatever csp is set for the application globally, which is usually not enough. if your api endpoint and your swagger docs are on different paths that is more noticeable. to be precise the root of the problem is that grape-swagger-rails just renders html pages and doesn't provide a mechanism to set a custom header. so, that means we're in control. we have to use rails features to achieve this.

my first tango with this problem was back in, i think it was 2018? i had a monolithic rails app and the swagger docs were acting up in the strangest ways, mostly on production. turns out, it was csp, the default policy was way too strict for swagger to load all its javascript and css. the app itself was working fine, but the docs were a mess of console errors and broken ui elements. i was pulling my hair out for a solid afternoon, until i realized i could override the headers on the controller level and this saved my life that day.

i've learned a few tricks since. the most common way to manage this is to add custom logic to your rails controller that renders the swagger page. this lets us define our csp just for that specific path, while your main application can use a different, potentially more restrictive csp. it’s about scoping those rules.

here is a basic idea of how you could implement it:

```ruby
class SwaggerUiController < ActionController::Base
  def index
    response.set_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';")

    render file: "#{Rails.root}/public/swagger/index.html", layout: false
  end
end
```

here’s the breakdown:

first i'm creating a controller that will be responsible for rendering the swagger page. in the `index` action, we are setting the `Content-Security-Policy` header using `response.set_header`. then `render` is used to render the `index.html` of our swagger page.

a word of caution here: `'unsafe-eval'` and `'unsafe-inline'` are generally discouraged in a production environment due to security concerns (specially `unsafe-eval`). however, in many cases, they are necessary for swagger-ui to function properly, at least with older versions or more customized setups. the snippet above is just an example, depending on your swagger-ui setup, you might need to adjust it or use a nonce to make it more secure.

so, after setting up this controller you need to route to this controller, the most basic way is through the rails routes:

```ruby
Rails.application.routes.draw do
  get '/swagger', to: 'swagger_ui#index'
end
```

this is the most simple example and you should adapt to your needs and current routes. the controller path `/swagger` should point to your generated swagger docs. in grape, you need to set the path that generates the documentation to something like `/swagger`. this snippet should point you to the right direction.

now, let's suppose you need to handle it more gracefully and you want to load your swagger docs from a gem or something different, and you can't just use the render function. you can set your csp rules in a `before_action` filter, like this:

```ruby
class SwaggerUiController < ActionController::Base
  before_action :set_swagger_csp

  def index
    # this could render the content from a gem
    render "my_swagger_docs"
  end

  private

  def set_swagger_csp
    response.set_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';")
  end
end
```

in this case, i'm using `before_action` to run the `set_swagger_csp` before the `index` action, so we are always setting that header before rendering the page. this is a more structured approach, specially if you have more actions in this controller.

it is important to remember that you have to include the `public/swagger/index.html` path in your `config.assets.paths` in your `config/application.rb` file. otherwise it will not be found by the render method.

the whole problem here stems from grape-swagger-rails lack of built-in features for csp customization and the challenge of dealing with the peculiarities of the swagger ui, specially with its javascript dependencies and sometimes very aggressive styles.

after trying the above solutions, in a very specific past project, i had to go a little further. you might think a more robust solution is to use rack middleware, and you're totally on track. although rack middleware is usually used for application level configurations, we can also use it to target specific paths, here is an example:

```ruby
# config/middleware/swagger_csp.rb
class SwaggerCsp
  def initialize(app)
    @app = app
  end

  def call(env)
    if env['PATH_INFO'] == '/swagger'
      status, headers, body = @app.call(env)
      headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';"
      [status, headers, body]
    else
      @app.call(env)
    end
  end
end
```

and then, in your `config/application.rb` or in a file in `config/initializers`, we use:

```ruby
config.middleware.use SwaggerCsp
```

the middleware intercepts every request, checks if it is on `/swagger`, and if it is, it sets the custom csp header. otherwise, it lets the request flow as normal, which makes this approach more generic and it avoids doing the same logic on the controllers.

the nice part about this is that now we don’t need to manage the csp logic at the controller level anymore, it’s handled by the middleware. this is very useful if you need to set the csp in many different swagger endpoints.

one caveat, if your swagger docs are not in the root level, `/swagger`, you have to change the condition `env['PATH_INFO'] == '/swagger'` to match your current swagger path, otherwise it will not work.

one time i was explaining all this to a junior dev and he looked at me dead serious and asked: "so, the csp is like a bouncer for my browser's resources?". i almost lost it. sometimes, the most straightforward analogies are the most hilarious.

now, when it comes to learning more about csp i highly recommend reading “high performance browser networking” by ilya grigorik, it has a solid explanation of security headers and performance optimizations. the mdn web docs also have a good, very complete, documentation about csp and other http headers. specifically the content security policy documentation is essential.

also, always remember that security is not an afterthought. testing and refining your csp is crucial to prevent both security vulnerabilities and functionality regressions. start with a restrictive policy and slowly add specific directives as needed. it's a fine line between security and functionality, but it's a line you have to walk carefully.

i think that’s all that i can say for now.
